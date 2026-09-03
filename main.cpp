#include <opencv2/geometry.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <malloc.h>
#include <linux/limits.h>
#include <unistd.h>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if CV_VERSION_MAJOR < 5
#error "requires OpenCV 5 or newer: aruco lives in objdetect"
#endif

namespace {

/**
 * Markers along one edge of the printed board.
 *
 * The board is a 5x5 grid with the centre omitted, so 24 markers carry the
 * pose. Changing this alone is not enough: the printed PDF, the marker ids
 * and `MODULES_ACROSS` all describe the same sheet and have to agree.
 */
constexpr int MARKERS_PER_SIDE = 5;

/**
 * Dictionary id of the board's top-left marker.
 *
 * Ids run consecutively from here in reading order, which is what lets a
 * detected id be turned back into a board coordinate by arithmetic alone.
 */
constexpr int FIRST_MARKER_ID = 5;

/**
 * The id the centre of the grid would have carried.
 *
 * That square is left blank so the spinning arm's axle has somewhere to sit
 * without covering a marker. It is skipped when the board model is built.
 */
constexpr int EMPTY_CENTER_ID = 17;

/**
 * ArUco modules across the printed sheet, markers and quiet zones together.
 *
 * The physical size divided by this gives the module pitch, which is how the
 * cm-space marker layout below is derived from one measurement of the sheet.
 */
constexpr int MODULES_ACROSS = 34;

/**
 * Printed width of the marker sheet, in centimetres.
 *
 * Measured from the sheet rather than taken from the page size, because A0
 * paper and the printer's margins do not agree to the millimetre and every
 * distance in board space is scaled by this.
 */
constexpr double BOARD_WIDTH_CM = 79.5;

/**
 * Printed height of the marker sheet, in centimetres.
 *
 * Not equal to the width: the same nominal square comes off the printer
 * slightly anisotropic, and keeping the two apart is what stops that error
 * from leaking into the fitted orbit radius.
 */
constexpr double BOARD_HEIGHT_CM = 79.3;

/** Horizontal size of one ArUco module, in centimetres. */
constexpr double MODULE_X_CM = BOARD_WIDTH_CM / MODULES_ACROSS;

/** Vertical size of one ArUco module, in centimetres. */
constexpr double MODULE_Y_CM = BOARD_HEIGHT_CM / MODULES_ACROSS;

/** Marker width in centimetres; a `DICT_4X4` marker is six modules across. */
constexpr double MARKER_WIDTH_CM = 6.0 * MODULE_X_CM;

/** Marker height in centimetres, six modules like the width. */
constexpr double MARKER_HEIGHT_CM = 6.0 * MODULE_Y_CM;

/** Centre-to-centre spacing in x: six modules of marker plus one of gap. */
constexpr double PITCH_X_CM = 7.0 * MODULE_X_CM;

/** Centre-to-centre spacing in y, matching `PITCH_X_CM`. */
constexpr double PITCH_Y_CM = 7.0 * MODULE_Y_CM;

/**
 * Half the board width, the x offset that puts the origin at its centre.
 *
 * Board space is centred on the axle rather than on a corner, so the diode's
 * orbit is a circle about the origin and its angle needs no further offset.
 */
constexpr double HALF_WIDTH_CM = BOARD_WIDTH_CM / 2.0;

/** Half the board height, the matching y offset for a centred origin. */
constexpr double HALF_HEIGHT_CM = BOARD_HEIGHT_CM / 2.0;

/**
 * Height of the LED above the printed sheet, in centimetres.
 *
 * The homography maps the board plane, not the plane the LED orbits in, so a
 * blob seen off-axis lands slightly outside its true radius. This is what the
 * correction uses to project it back down.
 */
constexpr double DIODE_HEIGHT_CM = 4.0;

/**
 * Dictionary the printed markers were generated from.
 *
 * 4x4 keeps the modules large enough to survive the robot camera's downscaled
 * preview, where a denser dictionary stops decoding first.
 */
constexpr int ARUCO_DICTIONARY = cv::aruco::DICT_4X4_50;

/**
 * Slack added around a view's marker bounds before searching it, in pixels.
 *
 * The crop follows the markers, but the diode orbits outside them and the
 * board drifts between frames. Too little padding and the LED leaves the
 * search window on a fast take; the cost of too much is only search time.
 */
constexpr int CROP_PADDING_PX = 260;

/**
 * Extra margin around a tracked board, in multiples of its marker pitch.
 *
 * Expressed in markers rather than pixels so it scales with however large the
 * board appears in that view: the robot preview is a quarter the size of the
 * passthrough and needs the same proportional margin, not the same one.
 */
constexpr double TRACK_MARGIN_MARKERS = 0.5;

/**
 * Factor the frame is shrunk by for the initial view-finding scan.
 *
 * Locating the boards only needs to be approximate, and detection on a
 * quarter-size image is roughly an order of magnitude cheaper. The full
 * resolution frame is used once the crops are known.
 */
constexpr int CROP_SCAN_DOWNSCALE = 4;

/**
 * Time constant of the crop's exponential smoother, in seconds.
 *
 * Converted to a per-frame alpha against the clip's own fps, so the crop
 * settles at the same rate in wall-clock terms whatever the recording ran at.
 * It is deliberately slow: a crop that chases per-frame jitter makes the
 * overlay panels shimmer.
 */
constexpr double CROP_SMOOTHING_SECONDS = 1.0;

/** Fraction the smoothed crop is grown by, so the board never touches its edge. */
constexpr double CROP_SLACK = 0.12;

/**
 * Time skipped at each end of the clip, in seconds.
 *
 * Both ends: a recording usually starts with the operator still settling and
 * ends with them reaching for the stop, and neither stretch is the steady
 * rotation the fit assumes.
 */
constexpr double WARMUP_SECONDS = 0.5;

/**
 * Seed forced into OpenCV's global RNG before the run.
 *
 * `findHomography` with RANSAC samples that generator, so without a fixed
 * seed the same clip yields slightly different numbers on every run. Pinning
 * it is what makes a reported latency reproducible.
 */
constexpr uint64_t RANDOM_SEED = 0;

/**
 * Radius the LED orbits at, in centimetres.
 *
 * Only a starting point for the annular search band; the radius actually used
 * is fitted from the track, so a differently built arm still measures.
 */
constexpr double DIODE_ORBIT_RADIUS_CM = 30.0;

/**
 * Width of the annulus the LED is searched in, in centimetres.
 *
 * Tied to the marker size rather than chosen outright, so a board printed at
 * a different scale widens the band by the same factor and the search keeps
 * the same tolerance in board units.
 */
constexpr double DIODE_BAND_WIDTH_CM = MARKER_WIDTH_CM;

/** Inner edge of the search annulus. */
constexpr double DIODE_INNER_RADIUS_CM =
    DIODE_ORBIT_RADIUS_CM - DIODE_BAND_WIDTH_CM / 2.0;

/** Outer edge of the search annulus. */
constexpr double DIODE_OUTER_RADIUS_CM =
    DIODE_ORBIT_RADIUS_CM + DIODE_BAND_WIDTH_CM / 2.0;

/**
 * Reach of the compass star's cardinal arms, in centimetres.
 *
 * The star is decoration drawn in board space, so sizing it from the orbit
 * radius keeps it in proportion to the board in every view without a
 * per-view constant.
 */
constexpr double COMPASS_POINT_CM = DIODE_ORBIT_RADIUS_CM * 2.0;
constexpr double COMPASS_DIAGONAL_CM =
    COMPASS_POINT_CM * 0.62;  ///< reach of the four diagonal arms
constexpr double COMPASS_OUTER_RING_CM =
    COMPASS_POINT_CM * 0.68;  ///< radius of the outer ring
constexpr double COMPASS_INNER_RING_CM =
    COMPASS_POINT_CM * 0.20;  ///< radius of the inner ring
constexpr double COMPASS_SHOULDER_CM =
    COMPASS_POINT_CM * 0.11;  ///< where an arm's flanks meet its neighbours
constexpr double COMPASS_LABEL_CM =
    COMPASS_POINT_CM * 1.14;  ///< radius the N/E/S/W labels sit at
constexpr double COMPASS_LABEL_HEIGHT_CM =
    COMPASS_POINT_CM * 0.19;  ///< label cap height in board units

/**
 * Weight the compass star is blended into the panel at.
 *
 * Drawn onto a copy and blended back rather than painted directly, so the
 * board stays readable underneath it and the star never hides a marker the
 * reader is trying to check.
 */
constexpr double COMPASS_OPACITY = 0.55;

/**
 * Vertical placement of the top view's band, as a fraction of the panel.
 *
 * The band panel shows each view's annulus at a fixed spot rather than where
 * the board happens to be, so the three views can be compared side by side.
 */
constexpr double BAND_TOP_CENTRE_Y = 1.0 / 3.0;
constexpr double BAND_BOTTOM_CENTRE_Y =
    3.0 / 4.0;  ///< vertical placement of the bottom views' bands
constexpr double BAND_TOP_DIAMETER =
    1.0 / 3.0;  ///< top view's band diameter, as a fraction of the panel
constexpr double BAND_BOTTOM_DIAMETER =
    1.0 / 5.0;  ///< bottom views' band diameter; two must fit side by side

constexpr int COMPASS_POINTS = 8;  ///< four cardinal arms and four diagonals
constexpr int COMPASS_NORTH_STEP = 0;  ///< which arm is coloured as north
constexpr int COMPASS_SOUTH_STEP = 4;  ///< which arm is coloured as south

/**
 * Smallest cap height a compass label is drawn at, in pixels.
 *
 * Below this the glyphs are illegible smudges, so the labels are dropped
 * rather than drawn: the robot preview is small enough for that to happen.
 */
constexpr int COMPASS_LABEL_MIN_PX = 7;

/**
 * Lowest redness score accepted as the LED, on a 0-255 scale.
 *
 * The first pass looks for the diode by colour. A frame whose peak falls
 * short is not treated as a weak detection but as none at all, and the
 * brightness search runs instead.
 */
constexpr int RED_LED_MIN_SCORE = 25;

/**
 * Chroma below which a view is treated as greyscale.
 *
 * The robot cameras deliver infrared, so their preview carries no colour and
 * the red-LED search cannot work there. Measuring the chroma decides which of
 * the two searches a view gets, rather than hard-coding which view is which.
 */
constexpr double GRAY_CHROMA_LIMIT = 6.0;

/**
 * Brightness cut-offs tried in turn when looking for the LED in a grey view.
 *
 * Descending, and the first that yields a blob of plausible area wins. A
 * single threshold cannot serve both a bright passthrough and a dim infrared
 * preview; walking down from the brightest keeps the strongest candidate
 * rather than the largest.
 */
const std::array<int, 7> BRIGHT_THRESHOLDS{235, 220, 205, 190, 175, 160, 145};
constexpr int BRIGHT_MIN_AREA = 2;  ///< smaller blobs are sensor noise
constexpr int BRIGHT_MAX_AREA =
    1500;  ///< larger blobs are a lamp or a reflection, not the LED

/**
 * Weight a new frame's board pose is mixed in at.
 *
 * The pose is re-fitted every frame and would otherwise jitter by a pixel or
 * two, which is visible in the overlay. Only the drawing is smoothed; the
 * angles the measurement is built from come from the unsmoothed fit.
 */
constexpr double POSE_SMOOTHING = 0.1;
constexpr int TABLE_COLUMNS = 9;  ///< columns in the results table panel

/**
 * Fractional bits used by the drawing calls.
 *
 * OpenCV takes integer coordinates, so lines drawn from board space would
 * otherwise snap to whole pixels and the fitted circle would look polygonal.
 * Shifting gives 1/16th-pixel placement.
 */
constexpr int DRAW_SHIFT = 4;
constexpr int PANEL_SIZE =
    720;  ///< edge of one panel in the output grid, in pixels
constexpr int OUTPUT_WIDTH = PANEL_SIZE * 2;   ///< two panels across
constexpr int OUTPUT_HEIGHT = PANEL_SIZE * 2;  ///< two panels down

/**
 * Overlay palette, in OpenCV's BGR order.
 *
 * `VIEW_COLORS` picks one per view so a reader can tell at a glance which
 * board a drawing belongs to; the rest label fixed parts of the overlay.
 */
const cv::Scalar GREEN{0, 255, 0};
const cv::Scalar GRID{130, 170, 120};   ///< the projected marker grid
const cv::Scalar YELLOW{0, 255, 255};   ///< the fitted orbit circle
const cv::Scalar RED{0, 0, 255};        ///< view 0, and the diode marker
const cv::Scalar MAGENTA{255, 0, 255};  ///< the detected diode blob
const cv::Scalar SITE_GREEN{
    64,
    111,
    46
};  ///< the brand green the numbers panel is painted in
const cv::Scalar WHITE{255, 255, 255};  ///< text
const cv::Scalar ORANGE{0, 165, 255};   ///< warnings in the results table
const cv::Scalar BLUE{255, 160, 0};     ///< view 2
const cv::Scalar COMPASS_LIGHT{232, 232, 232};  ///< lit facet of a compass arm
const cv::Scalar COMPASS_DARK{54, 54, 54};  ///< shaded facet of a compass arm
const cv::Scalar COMPASS_NORTH_LIGHT{
    58,
    58,
    236
};  ///< lit facet of the north arm
const cv::Scalar COMPASS_NORTH_DARK{
    34,
    34,
    118
};  ///< shaded facet of the north arm
const cv::Scalar COMPASS_SOUTH_LIGHT{
    236,
    146,
    58
};  ///< lit facet of the south arm
const cv::Scalar COMPASS_SOUTH_DARK{
    124,
    74,
    30
};  ///< shaded facet of the south arm
const std::array<const char*, 4> COMPASS_LABELS{
    "N",
    "E",
    "S",
    "W"
};  ///< cardinals, in compass-step order

/** One colour per view, so every drawing says which board it came from. */
const std::array<cv::Scalar, 3> VIEW_COLORS{RED, GREEN, BLUE};

/** Those colours by name, for the results table's legend column. */
const std::array<const char*, 3> VIEW_COLOR_NAMES{"red", "green", "blue"};

/**
 * Overlay font, relative to the executable rather than the process's cwd.
 *
 * Resolved against the binary's own directory and its parent, so the tool
 * runs from anywhere as long as `assets/` travels with it.
 */
const char* FONT_PATH = "assets/JetBrainsMono.ttf";

/**
 * Body text size on the reference 1920px-wide page, in points.
 *
 * The panels are laid out at 1920 and rendered at `PANEL_SIZE`, so sizes are
 * written once against the reference page and scaled on the way out.
 */
constexpr double PAGE_SIZE_TEXT = 17.0;
constexpr double PAGE_SIZE_DISPLAY =
    88.0;  ///< display size on the same reference page
constexpr double PAGE_TRACKING =
    0.18;  ///< letter spacing of the wordmark, as a fraction of cap height

/** Body text cap height in output pixels, scaled from the reference page. */
constexpr int TEXT_HEIGHT =
    static_cast<int>(PAGE_SIZE_TEXT * 1920 / PANEL_SIZE);

/** Wordmark cap height, in the same ratio to the body text as on the page. */
constexpr int LOGO_HEIGHT =
    static_cast<int>(TEXT_HEIGHT * (PAGE_SIZE_DISPLAY / PAGE_SIZE_TEXT));

constexpr int TEXT_LINE =
    TEXT_HEIGHT * 3 / 2;                      ///< baseline-to-baseline spacing
constexpr int TEXT_MARGIN = TEXT_HEIGHT * 2;  ///< panel padding
const char* LOGO_TEXT = "rhoyn";  ///< wordmark drawn on the numbers panel

/**
 * Index of the headset passthrough among the views.
 *
 * The passthrough is the reference every latency is measured against, and
 * `cluster_boards` orders views topmost first, which is what puts it here.
 */
constexpr int TOP_VIEW = 0;

/**
 * Number of board views in the recording.
 *
 * Not constant: `configure_views` sets it once the first frames reveal
 * whether the robot camera is mono or stereo, so neither case needs a flag.
 */
int VIEW_COUNT = 3;
bool TWO_BOTTOM_BOARDS = true;  ///< true when the robot camera is a stereo pair

/** View labels for the overlay and the results table, in view order. */
std::vector<const char*> VIEW_NAMES{
    "headset passthrough",
    "robot camera L",
    "robot camera R"
};

/** Views that are robot cameras, i.e. everything but the passthrough. */
std::vector<int> BOTTOM_VIEWS{1, 2};

/** Stands for a measurement a frame did not produce. */
const double NaN = std::numeric_limits<double>::quiet_NaN();
std::string INPUT_PATH = "input.mp4";  ///< recording to read, from `--input`
std::string OUTPUT_PATH =
    "output.mp4";  ///< four-panel render to write, from `--output`

/**
 * Frames the encoder thread may hold before the analysis has to wait.
 *
 * Encoding is slower than analysis, so a queue keeps both busy; a bound is
 * what stops a long clip from growing the queue until it exhausts memory.
 */
constexpr size_t WRITER_QUEUE_FRAMES = 8;

/**
 * One board as this frame saw it.
 *
 * The homography is stored image-to-board because that is the direction the
 * measurement needs: a diode found in pixels becomes a position in
 * centimetres. Drawing inverts it.
 */
struct ViewGeometry {
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  cv::Mat image_to_board;
};

/** Every view's geometry for one frame, indexed by view. */
using FrameGeometry = std::vector<ViewGeometry>;

/**
 * Where the LED was found in one view, in both spaces.
 *
 * The pixel track is kept alongside the board-space one so the overlay can
 * mark the raw detection rather than a reprojection of the fitted value.
 */
struct DiodeTrack {
  std::vector<cv::Point2d> pixels;
  std::vector<cv::Point2d> positions_cm;
};

/** A fitted orbit: centre and radius, in board centimetres. */
struct Circle {
  cv::Point2d center;
  double radius = 0.0;
};

/**
 * One view's straight-line fit of unwrapped angle against time.
 *
 * The slope is the angular speed and the intercept the phase, and it is the
 * difference between two views' intercepts that becomes a latency. `r2` and
 * `rmse` are carried for the table so a poor fit is visible rather than
 * silently averaged in.
 */
struct RegressionFit {
  double slope = 0.0;
  double intercept = 0.0;
  double r2 = 0.0;
  double rmse = 0.0;
};

/** The finished measurement: the shared speed, the fits, and the number. */
struct LatencyResult {
  double omega = 0.0;
  double latency_ms = 0.0;
  std::vector<RegressionFit> fits;
  int paired_frames = 0;
};

/**
 * The ArUco detector, the board model and the contrast equaliser.
 *
 * Built once and reused: constructing a detector per frame dominates the
 * per-frame cost, and CLAHE is what lets the dim infrared preview decode at
 * the same settings as the bright passthrough.
 */
struct Detector {
  cv::aruco::ArucoDetector aruco;
  cv::aruco::Board board;
  cv::Ptr<cv::CLAHE> clahe;
};

/** An inclusive pixel rectangle within the frame. */
struct CropBox {
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

/** What the container claims about the clip: rate, length and frame size. */
struct VideoInfo {
  double fps = 24.0;
  int frame_count = 0;
  cv::Size size;
};

/**
 * Exponentially smoothed crop for one view.
 *
 * Centre and side are smoothed rather than the four edges, so the window
 * keeps its shape while it follows the board instead of breathing.
 */
struct CropSmoother {
  bool initialised = false;
  double cx = 0.0;
  double cy = 0.0;
  double side = 0.0;
};

/**
 * Normal equations for a circle fit that never stores the samples.
 *
 * The orbit is fitted algebraically, so each new point only has to update a
 * 3x3 matrix and a vector. That keeps memory flat over a long clip and lets
 * the current best circle be read out at any frame for the overlay.
 */
struct RunningCircle {
  cv::Matx33d ata = cv::Matx33d::zeros();
  cv::Vec3d atb = cv::Vec3d::all(0.0);
  int count = 0;
  Circle circle;
};

/** Running sums for a least-squares line, again without keeping the samples. */
struct RunningRegression {
  double n = 0.0, sx = 0.0, sxx = 0.0, sy = 0.0, sxy = 0.0, syy = 0.0;
};

/**
 * State for turning wrapped angles into a continuous one.
 *
 * The measured angle lives in (-pi, pi] and the fit needs it to keep
 * climbing, so each step's wrap is detected and folded into a correction.
 */
struct RunningUnwrap {
  bool has_previous = false;
  double previous_raw = 0.0;
  double correction = 0.0;
};

/**
 * Which way the arm turns, decided once and then held.
 *
 * Locked on the first usable step rather than re-derived per frame: a single
 * noisy step near a wrap would otherwise flip the sign and unwrap the rest of
 * the clip backwards.
 */
struct RotationSign {
  bool locked = false;
  double value = 1.0;
};

/**
 * Encoder running on its own thread, with a bounded pool of frame buffers.
 *
 * Encoding costs more than analysing, so it is moved off the analysis thread.
 * Buffers are recycled through `spare` rather than reallocated per frame, and
 * `allocated` bounds the pool so a slow encoder applies backpressure instead
 * of consuming memory.
 */
struct BackgroundWriter {
  cv::VideoWriter video;
  std::thread encoder;
  std::mutex mutex;
  std::condition_variable frame_queued;
  std::condition_variable buffer_available;
  std::deque<cv::Mat> queued;
  std::vector<cv::Mat> spare;
  size_t allocated = 0;
  bool finishing = false;
};

/**
 * Everything carried from frame to frame, and the result at the end.
 *
 * The per-view vectors are parallel and all sized by `VIEW_COUNT`. Holding
 * the running fits here is what makes the pass single: no frame is revisited.
 */
struct Analysis {
  FrameGeometry geometry;
  int index = 0;
  double fps = 24.0;
  std::vector<DiodeTrack> diodes;
  std::vector<Circle> circles;
  std::vector<std::vector<double>> angles;
  std::vector<RunningCircle> running_circles;
  std::vector<RunningRegression> regressions;
  std::vector<RunningUnwrap> unwrappers;
  RotationSign rotation_sign;
  LatencyResult result;
  double first_time = 0.0;
  double last_time = 0.0;
  int paired = 0;
};

/** The four output panels, plus the full-size buffer the numbers are drawn in. */
struct Panels {
  cv::Mat stacked;
  cv::Mat raw;
  cv::Mat overlay;
  cv::Mat band;
  cv::Mat numbers;
  cv::Mat numbers_full;
};

/** A pen for laying out lines of text down a panel. */
struct TextCursor {
  cv::Mat* panel = nullptr;
  int x = 0;
  int y = 0;
  int height = TEXT_HEIGHT;
  int line = TEXT_LINE;
};

/** A float rectangle, used while accumulating marker extents. */
struct Bounds {
  float x0 = 0.0f, y0 = 0.0f, x1 = 0.0f, y1 = 0.0f;
};

/** Weighted running mean of a board's corners, for a steady overlay. */
struct PoseSmoother {
  std::vector<cv::Point2f> weighted;
  double weight = 0.0;
};

/**
 * Crop windows for every view, and whether all of them were found.
 *
 * `complete` matters during the opening scan: a frame that found only some of
 * the boards must not be allowed to fix the view count.
 */
struct ViewBoxes {
  std::vector<CropBox> bounds;
  bool complete = true;
};

/** How one view maps into its panel, and back into board space. */
struct PanelView {
  bool fitted = false;
  cv::Matx33d to_panel = cv::Matx33d::eye();
  cv::Mat to_board;
};

/** One `PanelView` per view, indexed alike. */
using PanelViews = std::vector<PanelView>;

/** The two tones an arm of the compass star is drawn with. */
struct CompassFacets {
  cv::Scalar shaded;
  cv::Scalar lit;
};

/** Markers detected in a frame: their quads, ids and centres. */
struct MarkerHits {
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  std::vector<cv::Point2d> centers;
};

/**
 * Size of a detected marker, as the side of a square with its area.
 *
 * Area rather than edge length, so a marker seen at an angle still reports a
 * sensible size. Used to normalise distances when clustering, so a large
 * board and a small one compete on equal terms.
 *
 * @param[in] quad  the marker's four image-space corners
 * @returns the equivalent square's side, in pixels
 * @exceptsafe basic
 */
double marker_scale(const std::vector<cv::Point2f>& quad) {
  return std::sqrt(std::abs(cv::contourArea(quad)));
}

/**
 * Locate the overlay font relative to the executable.
 *
 * Resolved from `/proc/self/exe` rather than the working directory, and the
 * binary's own directory is tried before its parent, so the tool runs from
 * anywhere as long as `assets/` travels with it.
 *
 * @returns absolute path to the font file
 * @throws std::runtime_error if it is in neither place
 * @exceptsafe basic
 */
std::string font_file() {
  char self[PATH_MAX];
  const ssize_t length = readlink("/proc/self/exe", self, sizeof(self) - 1);
  const std::filesystem::path binary(
      length > 0 ? std::string(self, length) : "."
  );
  const std::filesystem::path base = binary.parent_path();
  for (const std::filesystem::path& root : {base, base.parent_path()}) {
    const std::filesystem::path candidate = root / FONT_PATH;
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  throw std::runtime_error(
      "cannot find " + std::string(FONT_PATH) + " next to the executable"
  );
}

/**
 * The process-wide font, loaded on first use.
 *
 * Loading costs milliseconds and every panel draws text, so it is done once
 * and held. The reference is non-const because FreeType2's draw calls are
 * not const.
 *
 * @returns the shared font
 * @throws std::runtime_error if the font file cannot be found
 * @exceptsafe basic
 */
cv::Ptr<cv::freetype::FreeType2>& text_font() {
  static cv::Ptr<cv::freetype::FreeType2> font;
  if (font.empty()) {
    font = cv::freetype::createFreeType2();
    font->loadFontData(font_file(), 0);
  }
  return font;
}

/**
 * Group detected markers into one cluster per board.
 *
 * Every board carries the same marker ids, so a board can hold each id at
 * most once. Ids are consumed in order of how often they appear and each
 * detection is matched to the nearest cluster, with distance divided by that
 * cluster's marker size so a large board and a small one compete fairly.
 *
 * @param[in] hits    markers detected anywhere in the frame
 * @param[in] boards  number of clusters to produce
 * @returns one `MarkerHits` per board, ordered topmost first then left to
 *          right, which is what fixes view identity
 * @exceptsafe basic
 */
std::vector<MarkerHits> cluster_boards(
    const MarkerHits& hits,
    int boards
) {
  std::map<int, std::vector<size_t>> by_id;
  for (size_t i = 0; i < hits.ids.size(); ++i) by_id[hits.ids[i]].push_back(i);
  std::vector<std::pair<size_t, int>> ranked;
  for (const auto& entry : by_id)
    ranked.emplace_back(entry.second.size(), entry.first);
  std::sort(ranked.rbegin(), ranked.rend());

  std::vector<MarkerHits> clusters(boards);
  std::vector<cv::Point2d> centroid(boards);
  std::vector<double> scale(boards, 0.0);
  std::vector<int> members(boards, 0);

  for (const auto& rank : ranked) {
    const std::vector<size_t>& found = by_id[rank.second];
    std::vector<std::pair<double, std::pair<size_t, int>>> costs;
    for (size_t detection : found) {
      for (int c = 0; c < boards; ++c) {
        const cv::Point2d delta = hits.centers[detection] - centroid[c];
        const double reach = std::max(scale[c], 1.0);
        const double cost = members[c] == 0 ? 1e9 : cv::norm(delta) / reach;
        costs.push_back({cost, {detection, c}});
      }
    }
    std::sort(costs.begin(), costs.end());

    std::set<size_t> taken_detection;
    std::set<int> taken_cluster;
    for (const auto& entry : costs) {
      const size_t detection = entry.second.first;
      const int c = entry.second.second;
      if (taken_detection.count(detection) || taken_cluster.count(c)) continue;
      taken_detection.insert(detection);
      taken_cluster.insert(c);
      const double weight = members[c];
      centroid[c] =
          (centroid[c] * weight + hits.centers[detection]) / (weight + 1.0);
      const double size = marker_scale(hits.corners[detection]);
      scale[c] = (scale[c] * weight + size) / (weight + 1.0);
      ++members[c];
      clusters[c].ids.push_back(hits.ids[detection]);
      clusters[c].corners.push_back(hits.corners[detection]);
      clusters[c].centers.push_back(hits.centers[detection]);
    }
  }

  std::vector<int> order(boards);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return centroid[a].y < centroid[b].y;
  });
  std::sort(order.begin() + 1, order.end(), [&](int a, int b) {
    return centroid[a].x < centroid[b].x;
  });
  std::vector<MarkerHits> ordered;
  for (int c : order) ordered.push_back(std::move(clusters[c]));
  return ordered;
}

/**
 * Measure a string at a given cap height.
 *
 * Used to right-align and centre text, which the table and the wordmark both
 * need before they know where to start drawing.
 *
 * @param[in] text    string to measure
 * @param[in] height  cap height in pixels
 * @returns the advance width in pixels
 * @exceptsafe basic
 */
int text_width(
    const std::string& text,
    int height
) {
  int baseline = 0;
  return text_font()->getTextSize(text, height, -1, &baseline).width;
}

std::
    array<
        cv::Point2f,
        4>

    /**
 * Corners of one marker in board centimetres.
 *
 * The id encodes the position: ids run in reading order from
 * `FIRST_MARKER_ID`, so row and column follow by division. The origin is the
 * board's centre, which is where the arm's axle sits.
 *
 * @param[in] marker_id  dictionary id of the marker
 * @returns its four corners, clockwise from the top-left
 * @exceptsafe no-throw
 */
    marker_corners_cm(int marker_id) {
  const int index = marker_id - FIRST_MARKER_ID;
  const int row = index / MARKERS_PER_SIDE;
  const int col = index % MARKERS_PER_SIDE;
  const double x = col * PITCH_X_CM - HALF_WIDTH_CM;
  const double y = row * PITCH_Y_CM - HALF_HEIGHT_CM;
  const double w = MARKER_WIDTH_CM, h = MARKER_HEIGHT_CM;
  return {
      cv::Point2f(x, y),
      cv::Point2f(x + w, y),
      cv::Point2f(x + w, y + h),
      cv::Point2f(x, y + h)
  };
}

/**
 * Fit the board-to-image homography from all detected markers.
 *
 * Every corner of every marker is used, so 24 markers give 96
 * correspondences; the known board layout is what averages per-corner noise
 * down. RANSAC rejects the occasional mis-detected marker, and the RNG is
 * seeded so repeated runs agree.
 *
 * @param[in] corners  image-space quads, one per detected marker
 * @param[in] ids      marker id matching each quad
 * @returns the image-to-board homography, or an empty matrix if fewer than
 *          four markers were given or no homography could be found
 * @exceptsafe basic
 */
cv::Mat fit_board(
    const std::vector<std::vector<cv::Point2f>>& corners,
    const std::vector<int>& ids
) {
  if (ids.size() < 4) return cv::Mat();
  std::vector<cv::Point2f> image_points, board_points;
  for (size_t i = 0; i < ids.size(); ++i) {
    for (const auto& p : corners[i]) image_points.push_back(p);
    for (const auto& p : marker_corners_cm(ids[i])) board_points.push_back(p);
  }
  const cv::Mat board_to_image =
      cv::findHomography(board_points, image_points, cv::RANSAC, 3.0);
  if (board_to_image.empty()) return cv::Mat();
  return board_to_image.inv();
}

/**
 * Centre of a crop box.
 *
 * @param[in] box  the rectangle
 * @returns its centre, in the same pixel coordinates
 * @exceptsafe no-throw
 */
cv::Point2d box_centre(const CropBox& box) {
  return cv::Point2d(0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1));
}

/**
 * Draw one line of white text at a baseline position.
 *
 * @param[in,out] panel   image to draw into
 * @param[in]     text    string to draw
 * @param[in]     x       left edge, in pixels
 * @param[in]     y       baseline, in pixels
 * @param[in]     height  cap height, defaulting to the body size
 * @exceptsafe basic
 */
void draw_text(
    cv::Mat& panel,
    const std::string& text,
    int x,
    int y,
    int height = TEXT_HEIGHT
) {
  text_font()->putText(
      panel,
      text,
      cv::Point(x, y),
      height,
      WHITE,
      -1,
      cv::LINE_AA,
      false
  );
}

/**
 * Centroid of a marker's four corners.
 *
 * @param[in] quad  four corners
 * @returns their mean position
 * @exceptsafe no-throw
 */
cv::Point2d quad_centre(const std::vector<cv::Point2f>& quad) {
  cv::Point2d centre(0.0, 0.0);
  for (const auto& p : quad) centre += cv::Point2d(p.x, p.y);
  return centre / 4.0;
}

/**
 * Sample a circle as a closed polyline.
 *
 * Drawn as a polygon rather than with `cv::circle` because the circle lives
 * in board space and has to go through the homography, which turns it into
 * an ellipse the circle primitive cannot express.
 *
 * @param[in] center  centre, in whatever space the caller is working in
 * @param[in] radius  radius in the same units
 * @param[in] count   samples around the circle
 * @returns the sampled points, first and last coincident
 * @exceptsafe basic
 */
std::vector<cv::Point2f> orbit_points(
    const cv::Point2d& center,
    double radius,
    int count = 180
) {
  std::vector<cv::Point2f> out;
  out.reserve(count);
  for (int i = 0; i < count; ++i) {
    const double angle = 2.0 * CV_PI * i / (count - 1);
    out.emplace_back(
        center.x + radius * std::cos(angle),
        center.y + radius * std::sin(angle)
    );
  }
  return out;
}

/**
 * Fold an angle into (-pi, pi].
 *
 * @param[in] angle  angle in radians
 * @returns the equivalent angle in the principal range
 * @exceptsafe no-throw
 */
double wrap(double angle) {
  double value = std::fmod(angle + CV_PI, 2.0 * CV_PI);
  if (value < 0.0) value += 2.0 * CV_PI;
  return value - CV_PI;
}

/**
 * Apply a homography to one point.
 *
 * @param[in] m  3x3 transform
 * @param[in] p  point to map
 * @returns the mapped point, dehomogenised
 * @exceptsafe no-throw
 */
cv::Point2d apply_transform(
    const cv::Matx33d& m,
    const cv::Point2d& p
) {
  const cv::Vec3d out = m * cv::Vec3d(p.x, p.y, 1.0);
  return cv::Point2d(out[0] / out[2], out[1] / out[2]);
}

/**
 * Close out a running least-squares line.
 *
 * Everything comes from the accumulated sums, so the samples never have to be
 * stored. A near-zero denominator means the times were effectively identical
 * and the fit is refused rather than returned as a huge slope.
 *
 * @param[in] state  accumulated sums
 * @returns slope, intercept, r2 and rmse, all zero if the fit was refused
 * @exceptsafe no-throw
 */
RegressionFit regression_fit(const RunningRegression& state) {
  RegressionFit result;
  if (state.n < 2.0) return result;
  const double denominator = state.n * state.sxx - state.sx * state.sx;
  if (std::abs(denominator) < 1e-12) return result;
  result.slope = (state.n * state.sxy - state.sx * state.sy) / denominator;
  result.intercept = (state.sy - result.slope * state.sx) / state.n;
  const double ss_tot = state.syy - state.sy * state.sy / state.n;
  double ss_res =
      state.syy - result.intercept * state.sy - result.slope * state.sxy;
  if (ss_res < 0.0) ss_res = 0.0;
  result.r2 = ss_tot > 0.0 ? 1.0 - ss_res / ss_tot : 0.0;
  result.rmse = std::sqrt(ss_res / state.n);
  return result;
}

/**
 * Fix the arm's direction of travel from the first usable step.
 *
 * Decided once and then held. Re-deriving it per frame would let a single
 * noisy step near a wrap flip the sign and unwrap the rest of the clip in the
 * wrong direction.
 *
 * @param[in,out] rotation       the sign, updated only while unlocked
 * @param[in]     wrapped_delta  the step that decides it
 * @exceptsafe no-throw
 */
void lock_rotation_sign(
    RotationSign& rotation,
    double wrapped_delta
) {
  if (rotation.locked) return;
  rotation.locked = true;
  rotation.value = wrapped_delta >= 0.0 ? 1.0 : -1.0;
}

/**
 * True when both coordinates are finite.
 *
 * @param[in] p  point to test
 * @returns whether it can be used in arithmetic
 * @exceptsafe no-throw
 */
bool is_finite(const cv::Point2d& p) {
  return std::isfinite(p.x) && std::isfinite(p.y);
}

/**
 * Convert radians to turns, for the human-readable table.
 *
 * @param[in] radians  angle
 * @returns the same angle in revolutions
 * @exceptsafe no-throw
 */
double revolutions(double radians) { return radians / (2.0 * CV_PI); }

/**
 * The ids the printed board actually carries.
 *
 * The full run from `FIRST_MARKER_ID`, minus the centre square left open for
 * the axle. Built once on first use.
 *
 * @returns the id set, in ascending order
 * @exceptsafe basic
 */
const std::set<int>& marker_ids() {
  static const std::set<int> ids = [] {
    std::set<int> out;
    const int total = FIRST_MARKER_ID + MARKERS_PER_SIDE * MARKERS_PER_SIDE;
    for (int i = FIRST_MARKER_ID; i < total; ++i)
      if (i != EMPTY_CENTER_ID) out.insert(i);
    return out;
  }();
  return ids;
}

/**
 * Detector settings tuned for both views at once.
 *
 * One recording holds a bright passthrough and a dim, downscaled robot
 * preview, so the thresholds are deliberately permissive: a wide adaptive
 * window range, a low minimum perimeter so the small preview markers still
 * qualify, and a high error-correction rate. Sub-pixel corner refinement is
 * what the homography's accuracy rests on.
 *
 * @returns the parameter set
 * @exceptsafe basic
 */
cv::aruco::DetectorParameters make_detector_params() {
  cv::aruco::DetectorParameters params;
  params.adaptiveThreshWinSizeMin = 3;
  params.adaptiveThreshWinSizeMax = 45;
  params.adaptiveThreshWinSizeStep = 6;
  params.minMarkerPerimeterRate = 0.02;
  params.maxMarkerPerimeterRate = 4.0;
  params.polygonalApproxAccuracyRate = 0.06;
  params.minCornerDistanceRate = 0.03;
  params.minOtsuStdDev = 3.0;
  params.errorCorrectionRate = 0.9;
  params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  params.cornerRefinementWinSize = 5;
  return params;
}

/**
 * The dictionary the printed markers came from.
 *
 * @returns the predefined dictionary named by `ARUCO_DICTIONARY`
 * @exceptsafe basic
 */
cv::aruco::Dictionary make_dictionary() {
  return cv::aruco::getPredefinedDictionary(ARUCO_DICTIONARY);
}

/**
 * Build the board model from the known marker layout.
 *
 * Every marker is placed at z = 0 in board centimetres, so a pose recovered
 * against this model is already in the units the measurement works in.
 *
 * @returns the board
 * @exceptsafe basic
 */
cv::aruco::Board make_board() {
  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<int> ids;
  for (int marker_id : marker_ids()) {
    const auto flat = marker_corners_cm(marker_id);
    std::vector<cv::Point3f> quad;
    for (const auto& p : flat) quad.emplace_back(p.x, p.y, 0.0f);
    object_points.push_back(std::move(quad));
    ids.push_back(marker_id);
  }
  return cv::aruco::Board(object_points, make_dictionary(), ids);
}

/**
 * Map points from board centimetres into image pixels.
 *
 * The stored homography runs image-to-board because that is the direction the
 * measurement needs, so drawing inverts it here.
 *
 * @param[in] image_to_board  the view's homography
 * @param[in] points_cm       points in board space
 * @returns the same points in image space, empty if none were given
 * @exceptsafe basic
 */
std::vector<cv::Point2f> board_to_image_points(
    const cv::Mat& image_to_board,
    const std::vector<cv::Point2f>& points_cm
) {
  if (points_cm.empty()) return {};
  std::vector<cv::Point2f> out;
  cv::perspectiveTransform(points_cm, out, image_to_board.inv());
  return out;
}

/**
 * Project a board-space circle into the image as a polyline.
 *
 * @param[in] image_to_board  the view's homography
 * @param[in] centre          circle centre in board centimetres
 * @param[in] radius_cm       radius in centimetres
 * @returns the projected outline
 * @exceptsafe basic
 */
std::vector<cv::Point2f> board_circle_image(
    const cv::Mat& image_to_board,
    const cv::Point2d& centre,
    double radius_cm
) {
  return board_to_image_points(image_to_board, orbit_points(centre, radius_cm));
}

std::
    map<int,
        std::vector<cv::Point2f>>

    /**
 * Project every marker of the board model into the image.
 *
 * Drawn as the reference grid, which is what lets a reader see whether the
 * fitted pose actually agrees with the markers under it.
 *
 * @param[in] image_to_board  the view's homography
 * @returns each marker id mapped to its projected quad
 * @exceptsafe basic
 */
    project_all_markers(const cv::Mat& image_to_board) {
  std::map<int, std::vector<cv::Point2f>> projected;
  for (int marker_id : marker_ids()) {
    const auto flat = marker_corners_cm(marker_id);
    projected[marker_id] = board_to_image_points(
        image_to_board,
        {flat[0], flat[1], flat[2], flat[3]}
    );
  }
  return projected;
}

/**
 * Project the sheet's outline into the image.
 *
 * @param[in] image_to_board  the view's homography
 * @returns the four projected corners
 * @exceptsafe basic
 */
std::vector<cv::Point2f> project_board_outline(const cv::Mat& image_to_board) {
  const float hw = HALF_WIDTH_CM, hh = HALF_HEIGHT_CM;
  return board_to_image_points(
      image_to_board,
      {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}
  );
}

/**
 * Round points to whole pixels for the mask-filling calls.
 *
 * Masks are binary, so sub-pixel placement buys nothing there; the drawing
 * path uses `to_fixed` instead.
 *
 * @param[in] points  sub-pixel points
 * @returns the rounded points
 * @exceptsafe basic
 */
std::vector<cv::Point> to_int(const std::vector<cv::Point2f>& points) {
  std::vector<cv::Point> out;
  out.reserve(points.size());
  for (const auto& p : points) out.emplace_back(cvRound(p.x), cvRound(p.y));
  return out;
}

/**
 * Fill a polygon into a single-channel mask.
 *
 * @param[in,out] mask    image to fill into
 * @param[in]     points  polygon outline
 * @param[in]     value   value written inside it
 * @exceptsafe basic
 */
void fill_poly(
    cv::Mat& mask,
    const std::vector<cv::Point2f>& points,
    int value
) {
  cv::fillPoly(
      mask,
      std::vector<std::vector<cv::Point>>{to_int(points)},
      value
  );
}

/**
 * Axis-aligned extent of a point set.
 *
 * @param[in] points  at least one point
 * @returns the bounding rectangle
 * @exceptsafe no-throw
 */
Bounds bounds_of(const std::vector<cv::Point2f>& points) {
  Bounds bounds{points[0].x, points[0].y, points[0].x, points[0].y};
  for (const auto& p : points) {
    bounds.x0 = std::min(bounds.x0, p.x);
    bounds.y0 = std::min(bounds.y0, p.y);
    bounds.x1 = std::max(bounds.x1, p.x);
    bounds.y1 = std::max(bounds.y1, p.y);
  }
  return bounds;
}

/**
 * Detect this board design's markers anywhere in an image.
 *
 * @param[in]     gray      greyscale image to search
 * @param[in,out] detector  dictionary and tuned detector parameters
 * @returns the accepted markers with their centres, discarding ids the board
 *          does not carry
 * @exceptsafe basic
 */
MarkerHits detect_valid_markers(
    const cv::Mat& gray,
    Detector& detector
) {
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector.aruco.detectMarkers(gray, corners, ids);
  MarkerHits hits;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!marker_ids().count(ids[i])) continue;
    hits.corners.push_back(corners[i]);
    hits.ids.push_back(ids[i]);
    hits.centers.push_back(quad_centre(corners[i]));
  }
  return hits;
}

/**
 * Bounding box of every corner of every marker in a view.
 *
 * @param[in] corners  the view's marker quads
 * @returns the enclosing box, or a zero box if no markers were given
 * @exceptsafe basic
 */
CropBox corners_bounding_box(
    const std::vector<std::vector<cv::Point2f>>& corners
) {
  std::vector<cv::Point2f> points;
  for (const auto& quad : corners)
    for (const auto& p : quad) points.push_back(p);
  if (points.empty()) return CropBox{0, 0, 0, 0};
  const Bounds b = bounds_of(points);
  return CropBox{
      static_cast<int>(b.x0),
      static_cast<int>(b.y0),
      static_cast<int>(b.x1),
      static_cast<int>(b.y1)
  };
}

/**
 * Locate every board from scratch by scanning the whole frame.
 *
 * @param[in]     gray       greyscale frame to search
 * @param[in,out] detector   dictionary and tuned detector parameters
 * @param[in]     downscale  factor the frame is reduced by before detection
 * @returns bounds per view, with `complete` false when any view came back
 *          with too few markers to fit
 * @exceptsafe basic
 */
ViewBoxes scan_view_boxes(
    const cv::Mat& gray,
    Detector& detector,
    int downscale
) {
  const double scale = 1.0 / downscale;
  cv::Mat small;
  cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
  MarkerHits hits = detect_valid_markers(small, detector);
  small.release();
  for (auto& quad : hits.corners)
    for (auto& p : quad) p *= static_cast<float>(downscale);
  for (auto& centre : hits.centers) centre *= 1.0 * downscale;

  ViewBoxes scan;
  scan.bounds.assign(VIEW_COUNT, CropBox{0, 0, 0, 0});
  const std::vector<MarkerHits> clusters = cluster_boards(hits, VIEW_COUNT);
  for (int view = 0; view < VIEW_COUNT; ++view) {
    if (clusters[view].ids.size() < 4) {
      scan.complete = false;
      continue;
    }
    scan.bounds[view] = corners_bounding_box(clusters[view].corners);
  }
  return scan;
}

/**
 * Infer how many boards a frame holds from repeated ids.
 *
 * Every board carries the same ids, so an id seen three times means three
 * boards are in shot. This is what decides mono against stereo without a
 * flag, before any clustering has happened.
 *
 * @param[in] hits  markers detected anywhere in the frame
 * @returns the highest repeat count, i.e. the number of boards
 * @exceptsafe basic
 */
int boards_in_frame(const MarkerHits& hits) {
  std::map<int, int> seen;
  int boards = 0;
  for (int id : hits.ids) boards = std::max(boards, ++seen[id]);
  return boards;
}

/**
 * Grow a box by a fixed pixel margin, clamped to the frame.
 *
 * @param[in] box      box to grow
 * @param[in] padding  margin in pixels
 * @param[in] size     frame size the result is clamped to
 * @returns the padded box
 * @exceptsafe no-throw
 */
CropBox pad_box(
    const CropBox& box,
    int padding,
    const cv::Size& size
) {
  return CropBox{
      std::max(0, box.x0 - padding),
      std::max(0, box.y0 - padding),
      std::min(size.width, box.x1 + padding),
      std::min(size.height, box.y1 + padding)
  };
}

/**
 * Locate every board, retrying at full resolution when needed.
 *
 * The fast downscaled scan misses boards that are small or dim, so an
 * incomplete result is scanned again at full resolution.
 *
 * @param[in]     gray      greyscale frame to search
 * @param[in,out] detector  dictionary and tuned detector parameters
 * @returns bounds per view
 * @exceptsafe basic
 */
ViewBoxes view_crop_boxes(
    const cv::Mat& gray,
    Detector& detector
) {
  const ViewBoxes scan = scan_view_boxes(gray, detector, CROP_SCAN_DOWNSCALE);
  return scan.complete ? scan : scan_view_boxes(gray, detector, 1);
}

/**
 * Detect markers inside one crop of the frame.
 *
 * The crop is contrast-enhanced first, because the robot camera views are
 * small and washed out. Only the largest instance of each id is kept, since a
 * neighbouring board repeats the same ids.
 *
 * @param[in]     gray      greyscale frame the crop is taken from
 * @param[in]     box       region to search, in frame coordinates
 * @param[in,out] detector  dictionary, board model and detector parameters
 * @returns the markers found, in frame coordinates
 * @exceptsafe basic
 */
MarkerHits detect_markers_in_crop(
    const cv::Mat& gray,
    const CropBox& box,
    Detector& detector
) {
  cv::Mat enhanced;
  const cv::Range rows(box.y0, box.y1), columns(box.x0, box.x1);
  detector.clahe->apply(gray(rows, columns), enhanced);

  std::vector<std::vector<cv::Point2f>> corners, rejected;
  std::vector<int> ids;
  detector.aruco.detectMarkers(enhanced, corners, ids, rejected);
  if (corners.size() >= 3) {
    detector.aruco.refineDetectedMarkers(
        enhanced,
        detector.board,
        corners,
        ids,
        rejected
    );
  }

  const cv::Point2f offset(box.x0, box.y0);
  std::map<int, std::pair<std::vector<cv::Point2f>, double>> largest;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!marker_ids().count(ids[i])) continue;
    std::vector<cv::Point2f> shifted;
    for (const auto& p : corners[i]) shifted.push_back(p + offset);
    const double area = std::abs(cv::contourArea(shifted));
    const auto it = largest.find(ids[i]);
    if (it != largest.end() && area <= it->second.second) continue;
    largest[ids[i]] = {shifted, area};
  }

  MarkerHits hits;
  for (const auto& entry : largest) {
    hits.ids.push_back(entry.first);
    hits.corners.push_back(entry.second.first);
    hits.centers.push_back(quad_centre(entry.second.first));
  }
  return hits;
}

/**
 * Margin to keep around a tracked board, in pixels.
 *
 * Derived from how large the board appears rather than fixed, so the small
 * robot preview and the large passthrough get the same margin in board terms
 * and the crop behaves the same in both.
 *
 * @param[in] span  the board's larger side, in pixels
 * @returns the margin in pixels
 * @exceptsafe no-throw
 */
int track_margin(int span) {
  const double markers = TRACK_MARGIN_MARKERS * span;
  return static_cast<int>(markers * MARKER_WIDTH_CM / BOARD_WIDTH_CM);
}

/**
 * Grow a box by a margin scaled to its own size, clamped to the frame.
 *
 * @param[in] box   box to grow
 * @param[in] size  frame size the result is clamped to
 * @returns the grown box
 * @exceptsafe no-throw
 */
CropBox grow_box(
    const CropBox& box,
    const cv::Size& size
) {
  const int margin = track_margin(std::max(box.x1 - box.x0, box.y1 - box.y0));
  return CropBox{
      std::max(0, box.x0 - margin),
      std::max(0, box.y0 - margin),
      std::min(size.width, box.x1 + margin),
      std::min(size.height, box.y1 + margin)
  };
}

/**
 * Whether a box has enough area to crop from.
 *
 * A degenerate box reaches OpenCV as an empty ROI and throws, so this is
 * checked before any crop is taken.
 *
 * @param[in] box  the box
 * @returns true when it is at least two pixels on both sides
 * @exceptsafe no-throw
 */
bool box_is_usable(const CropBox& box) {
  return box.x1 - box.x0 > 1 && box.y1 - box.y0 > 1;
}

/**
 * Drop markers that belong to a neighbouring board.
 *
 * Search boxes grow between frames and can reach a second board, whose
 * identical ids would otherwise be fitted into this view.
 *
 * @param[in,out] hits     markers to filter in place
 * @param[in]     tracked  last known bounds of every view
 * @param[in]     view     view being filtered for
 * @exceptsafe basic
 */
void keep_nearest_to_view(
    MarkerHits& hits,
    const std::vector<CropBox>& tracked,
    int view
) {
  MarkerHits kept;
  for (size_t i = 0; i < hits.ids.size(); ++i) {
    int nearest = view;
    double best = std::numeric_limits<double>::infinity();
    for (int other = 0; other < VIEW_COUNT; ++other) {
      if (!box_is_usable(tracked[other])) continue;
      const cv::Point2d delta = hits.centers[i] - box_centre(tracked[other]);
      const double distance = delta.dot(delta);
      if (distance < best) {
        best = distance;
        nearest = other;
      }
    }
    if (nearest != view) continue;
    kept.ids.push_back(hits.ids[i]);
    kept.corners.push_back(hits.corners[i]);
    kept.centers.push_back(hits.centers[i]);
  }
  hits = std::move(kept);
}

/**
 * Turn one board's detections into a view, fitting its pose if it can.
 *
 * Fewer than four markers leaves the homography empty rather than fitting a
 * bad one; the caller treats an empty matrix as "this view is not measurable
 * in this frame" and carries on.
 *
 * @param[in] hits  one board's markers, consumed
 * @returns the view, its homography empty if too few markers were present
 * @exceptsafe basic
 */
ViewGeometry fit_view(MarkerHits hits) {
  ViewGeometry vg;
  vg.corners = std::move(hits.corners);
  vg.ids = std::move(hits.ids);
  if (vg.ids.size() >= 4) vg.image_to_board = fit_board(vg.corners, vg.ids);
  return vg;
}

/**
 * Format a number for the table, or "--" when there isn't one.
 *
 * Non-finite values are frequent and expected — a view that never resolved
 * has no slope — so they print as a dash rather than as "nan".
 *
 * @param[in] value     the number
 * @param[in] decimals  digits after the point
 * @returns the formatted string
 * @exceptsafe basic
 */
std::string number(
    double value,
    int decimals
) {
  if (!std::isfinite(value)) return "--";
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
  return buffer;
}

/**
 * Draw a printf-formatted line and advance the cursor.
 *
 * @param[in,out] cursor  pen position, moved down one line
 * @param[in]     format  printf format string
 * @exceptsafe basic
 */
void print_line(
    TextCursor& cursor,
    const char* format,
    ...
) {
  char buffer[512];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  draw_text(*cursor.panel, buffer, cursor.x, cursor.y, cursor.height);
  cursor.y += cursor.line;
}

/**
 * Draw the wordmark right-aligned above a baseline, glyph by glyph.
 *
 * Drawn one character at a time because the mark is letter-spaced and the
 * font's own advance would ignore the tracking. The total is measured first
 * so the result lands flush against the right edge.
 *
 * @param[in,out] panel   image drawn into
 * @param[in]     right   right edge to align to, in pixels
 * @param[in]     bottom  baseline, in pixels
 * @param[in]     height  cap height, in pixels
 * @exceptsafe basic
 */
void draw_logo(
    cv::Mat& panel,
    int right,
    int bottom,
    int height
) {
  const int tracking = static_cast<int>(std::lround(PAGE_TRACKING * height));
  const std::string logo(LOGO_TEXT);
  int total = -tracking;
  for (char glyph : logo)
    total += text_width(std::string(1, glyph), height) + tracking;
  int cursor = right - total;
  for (char glyph : logo) {
    const std::string one(1, glyph);
    draw_text(panel, one, cursor, bottom - height, height);
    cursor += text_width(one, height) + tracking;
  }
}

/**
 * Encoder loop: drain queued frames until the writer is finishing.
 *
 * Each buffer is returned to the spare pool after it is written and the
 * analysis thread is woken, which is what bounds memory: the pool is fixed
 * and a slow encoder makes the producer wait rather than allocate.
 *
 * @param[in,out] writer  the writer whose queue is drained
 * @exceptsafe basic
 */
void writer_encode_queued(BackgroundWriter& writer) {
  for (;;) {
    cv::Mat buffer;
    {
      std::unique_lock<std::mutex> lock(writer.mutex);
      writer.frame_queued.wait(lock, [&writer] {
        return !writer.queued.empty() || writer.finishing;
      });
      if (writer.queued.empty()) return;
      buffer = std::move(writer.queued.front());
      writer.queued.pop_front();
    }
    writer.video.write(buffer);
    {
      std::lock_guard<std::mutex> lock(writer.mutex);
      writer.spare.push_back(std::move(buffer));
    }
    writer.buffer_available.notify_one();
  }
}

/**
 * Map one pixel into board centimetres.
 *
 * @param[in] image_to_board  the view's homography
 * @param[in] pixel           point in image space
 * @returns the same point in board space
 * @exceptsafe basic
 */
cv::Point2d image_to_board_point(
    const cv::Mat& image_to_board,
    const cv::Point2d& pixel
) {
  std::vector<cv::Point2f> src{cv::Point2f(pixel.x, pixel.y)}, dst;
  cv::perspectiveTransform(src, dst, image_to_board);
  return cv::Point2d(dst[0].x, dst[0].y);
}

/**
 * Test whether a pixel lies on the diode's orbit.
 *
 * The test is made in board centimetres rather than pixels, so it holds from
 * any viewpoint.
 *
 * @param[in] image_to_board  pose of the board the pixel belongs to
 * @param[in] pixel           candidate position in frame coordinates
 * @returns true when the radius falls inside the orbit band
 * @exceptsafe basic
 */
bool radius_in_band(
    const cv::Mat& image_to_board,
    const cv::Point2d& centre,
    const cv::Point2d& pixel
) {
  const cv::Point2d board_point = image_to_board_point(image_to_board, pixel);
  const double radius =
      std::hypot(board_point.x - centre.x, board_point.y - centre.y);
  return radius >= DIODE_INNER_RADIUS_CM && radius <= DIODE_OUTER_RADIUS_CM;
}

/**
 * Scale points into the drawing calls' fixed-point coordinates.
 *
 * Everything drawn comes from a homography and would visibly stairstep if
 * rounded to whole pixels; `DRAW_SHIFT` buys sixteenths.
 *
 * @param[in] points  sub-pixel points
 * @returns the same points scaled by 2^DRAW_SHIFT
 * @exceptsafe basic
 */
std::vector<cv::Point> to_fixed(const std::vector<cv::Point2f>& points) {
  std::vector<cv::Point> out;
  out.reserve(points.size());
  const float unit = 1 << DRAW_SHIFT;
  for (const auto& p : points)
    out.emplace_back(cvRound(p.x * unit), cvRound(p.y * unit));
  return out;
}

/**
 * Draw a polyline at sub-pixel precision.
 *
 * @param[in,out] canvas  image drawn into
 * @param[in]     points  vertices
 * @param[in]     closed  whether to join the last vertex to the first
 * @param[in]     color   line colour
 * @param[in]     width   line width in pixels
 * @exceptsafe basic
 */
void draw_poly(
    cv::Mat& canvas,
    const std::vector<cv::Point2f>& points,
    bool closed,
    const cv::Scalar& color,
    int width
) {
  const std::vector<std::vector<cv::Point>> polygon{to_fixed(points)};
  cv::polylines(canvas, polygon, closed, color, width, cv::LINE_AA, DRAW_SHIFT);
}

/**
 * Draw one sub-pixel line segment.
 *
 * @param[in,out] canvas  image drawn into
 * @param[in]     from    start point
 * @param[in]     to      end point
 * @param[in]     color   line colour
 * @exceptsafe basic
 */
void draw_line(
    cv::Mat& canvas,
    const cv::Point2f& from,
    const cv::Point2f& to,
    const cv::Scalar& color
) {
  const std::vector<cv::Point> ends = to_fixed({from, to});
  cv::line(canvas, ends[0], ends[1], color, 1, cv::LINE_AA, DRAW_SHIFT);
}

/**
 * Draw a circle at sub-pixel precision.
 *
 * @param[in,out] canvas  image drawn into
 * @param[in]     centre  centre in pixels
 * @param[in]     radius  radius in whole pixels
 * @param[in]     color   colour
 * @param[in]     width   line width, negative to fill
 * @exceptsafe basic
 */
void draw_circle(
    cv::Mat& canvas,
    const cv::Point2f& centre,
    int radius,
    const cv::Scalar& color,
    int width
) {
  const std::vector<cv::Point> spot = to_fixed({centre});
  const int scaled = radius * (1 << DRAW_SHIFT);
  cv::circle(canvas, spot[0], scaled, color, width, cv::LINE_AA, DRAW_SHIFT);
}

/**
 * Draw text centred on a point rather than from its baseline.
 *
 * The compass labels are placed by their middles, so the string is measured
 * and the origin shifted by half its box.
 *
 * @param[in,out] canvas  image drawn into
 * @param[in]     text    string to draw
 * @param[in]     at      the point the text is centred on
 * @param[in]     height  cap height in pixels
 * @param[in]     color   text colour
 * @exceptsafe basic
 */
void draw_label(
    cv::Mat& canvas,
    const std::string& text,
    const cv::Point2f& at,
    int height,
    const cv::Scalar& color
) {
  int baseline = 0;
  const cv::Size box = text_font()->getTextSize(text, height, -1, &baseline);
  const cv::Point origin(
      cvRound(at.x - box.width / 2.0),
      cvRound(at.y - box.height / 2.0)
  );
  text_font()
      ->putText(canvas, text, origin, height, color, -1, cv::LINE_AA, false);
}

/**
 * Fill a polygon at sub-pixel precision.
 *
 * @param[in,out] canvas  image drawn into
 * @param[in]     points  outline
 * @param[in]     color   fill colour
 * @exceptsafe basic
 */
void fill_shape(
    cv::Mat& canvas,
    const std::vector<cv::Point2f>& points,
    const cv::Scalar& color
) {
  const std::vector<std::vector<cv::Point>> shape{to_fixed(points)};
  cv::fillPoly(canvas, shape, color, cv::LINE_AA, DRAW_SHIFT);
}

/**
 * Local scale of a view at a point, in pixels per board centimetre.
 *
 * Measured by projecting one-centimetre steps along both board axes and
 * averaging their image lengths, because the homography is perspective: the
 * scale is a property of where you are on the board, not of the view.
 *
 * @param[in] image_to_board  the view's homography
 * @param[in] centre          board point to measure at
 * @returns pixels per centimetre there
 * @exceptsafe basic
 */
double board_pixels_per_cm(
    const cv::Mat& image_to_board,
    const cv::Point2d& centre
) {
  const cv::Point2f origin(centre.x, centre.y);
  const std::vector<cv::Point2f> axes = board_to_image_points(
      image_to_board,
      {origin, origin + cv::Point2f(1, 0), origin + cv::Point2f(0, 1)}
  );
  return 0.5 * (cv::norm(axes[1] - axes[0]) + cv::norm(axes[2] - axes[0]));
}

/**
 * A point on the compass rose, in board centimetres.
 *
 * Step 0 points along negative y, which is the board's own up and the
 * direction diode angles are measured from, so the rose reads as a dial for
 * the phase rather than as loose decoration.
 *
 * @param[in] centre     rose centre in board centimetres
 * @param[in] step       eighth of a turn, clockwise from north
 * @param[in] radius_cm  distance from the centre
 * @returns the point in board space
 * @exceptsafe no-throw
 */
cv::Point2f compass_point_cm(
    const cv::Point2d& centre,
    int step,
    double radius_cm
) {
  const double angle = step * 2.0 * CV_PI / COMPASS_POINTS;
  return cv::Point2f(
      centre.x + radius_cm * std::sin(angle),
      centre.y - radius_cm * std::cos(angle)
  );
}

/**
 * How far an arm of the rose reaches: cardinals are longer than diagonals.
 *
 * @param[in] step  eighth of a turn
 * @returns the arm's length in board centimetres
 * @exceptsafe no-throw
 */
double compass_reach_cm(int step) {
  return step % 2 ? COMPASS_DIAGONAL_CM : COMPASS_POINT_CM;
}

/**
 * The two tones an arm is drawn with, north and south singled out.
 *
 * @param[in] step  eighth of a turn
 * @returns the shaded and lit colours for that arm
 * @exceptsafe basic
 */
CompassFacets compass_facets(int step) {
  if (step == COMPASS_NORTH_STEP)
    return {COMPASS_NORTH_DARK, COMPASS_NORTH_LIGHT};
  if (step == COMPASS_SOUTH_STEP)
    return {COMPASS_SOUTH_DARK, COMPASS_SOUTH_LIGHT};
  return {COMPASS_DARK, COMPASS_LIGHT};
}

/**
 * Draw the star and rings of a compass rose flat on the board.
 *
 * The rose is laid out in board centimetres and projected through the pose,
 * so it lies in the board plane and turns with it. North is the board's own
 * up, the direction the diode angles are measured from, which makes the rose
 * a dial the phase can be read against. Each of the eight points is split
 * along its axis into a shaded and a lit facet, so the star still reads as
 * solid where it crosses the dark markers, and the north and south points
 * carry the needle colours that name the axis at a glance.
 *
 * @param[in,out] canvas    panel drawn into
 * @param[in]     to_board  panel-to-board homography of the view
 * @param[in]     centre    orbit centre in board centimetres
 * @exceptsafe basic
 */
void draw_compass_rose(
    cv::Mat& canvas,
    const cv::Mat& to_board,
    const cv::Point2d& centre
) {
  for (double radius : {COMPASS_OUTER_RING_CM, COMPASS_INNER_RING_CM})
    draw_poly(
        canvas,
        board_to_image_points(to_board, orbit_points(centre, radius)),
        true,
        COMPASS_LIGHT,
        1
    );

  for (int step = 0; step < COMPASS_POINTS; ++step) {
    const std::vector<cv::Point2f> spike = board_to_image_points(
        to_board,
        {cv::Point2f(centre.x, centre.y),
         compass_point_cm(centre, step - 1, COMPASS_SHOULDER_CM),
         compass_point_cm(centre, step, compass_reach_cm(step)),
         compass_point_cm(centre, step + 1, COMPASS_SHOULDER_CM)}
    );
    const CompassFacets facets = compass_facets(step);
    fill_shape(canvas, {spike[0], spike[1], spike[2]}, facets.shaded);
    fill_shape(canvas, {spike[0], spike[2], spike[3]}, facets.lit);
    draw_poly(canvas, spike, true, facets.lit, 1);
  }
}

/**
 * Mark the four cardinal directions of a compass rose.
 *
 * Drawn apart from the rose so the letters stay opaque while the star behind
 * them is blended back, and skipped outright on a board too small in the
 * panel for the letters to resolve.
 *
 * @param[in,out] canvas    panel drawn into
 * @param[in]     to_board  panel-to-board homography of the view
 * @param[in]     centre    orbit centre in board centimetres
 * @exceptsafe basic
 */
void draw_compass_labels(
    cv::Mat& canvas,
    const cv::Mat& to_board,
    const cv::Point2d& centre
) {
  const double pixels_per_cm = board_pixels_per_cm(to_board, centre);
  const int height = cvRound(COMPASS_LABEL_HEIGHT_CM * pixels_per_cm);
  if (height < COMPASS_LABEL_MIN_PX) return;
  for (size_t cardinal = 0; cardinal < COMPASS_LABELS.size(); ++cardinal) {
    const int step = 2 * static_cast<int>(cardinal);
    const std::vector<cv::Point2f> at = board_to_image_points(
        to_board,
        {compass_point_cm(centre, step, COMPASS_LABEL_CM)}
    );
    const CompassFacets facets = compass_facets(step);
    draw_label(canvas, COMPASS_LABELS[cardinal], at[0], height, facets.lit);
  }
}

/**
 * Pixel rectangle enclosing a view's search annulus.
 *
 * The annulus is a circle in board space and therefore an ellipse in the
 * image, so its extent is found by projecting and bounding rather than by
 * arithmetic on a radius. Padded slightly and clipped to the frame.
 *
 * @param[in] image_to_board  the view's homography
 * @param[in] centre          orbit centre in board centimetres
 * @param[in] size            frame size to clip against
 * @returns the enclosing rectangle, the whole frame if nothing projected
 * @exceptsafe basic
 */
cv::Rect board_annulus_bbox(
    const cv::Mat& image_to_board,
    const cv::Point2d& centre,
    const cv::Size& size
) {
  const std::vector<cv::Point2f> outer =
      board_circle_image(image_to_board, centre, DIODE_OUTER_RADIUS_CM);
  const cv::Rect frame(0, 0, size.width, size.height);
  if (outer.empty()) return frame;
  const Bounds b = bounds_of(outer);
  const cv::Point top_left{
      static_cast<int>(b.x0) - 48,
      static_cast<int>(b.y0) - 48
  };
  const cv::Point bottom_right{
      static_cast<int>(b.x1) + 49,
      static_cast<int>(b.y1) + 49
  };
  return cv::Rect(top_left, bottom_right) & frame;
}

/**
 * Mask covering the ring the diode can be in.
 *
 * Filled outer circle with the inner one punched back out, both projected
 * from board space. Restricting the search to this ring is what keeps a lamp
 * or a reflection elsewhere in the room from being taken for the LED.
 *
 * @param[in] image_to_board  the view's homography
 * @param[in] centre          orbit centre in board centimetres
 * @param[in] size            mask size
 * @returns an 8-bit mask, 255 inside the annulus
 * @exceptsafe basic
 */
cv::Mat board_annulus_mask(
    const cv::Mat& image_to_board,
    const cv::Point2d& centre,
    const cv::Size& size
) {
  cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
  fill_poly(
      mask,
      board_circle_image(image_to_board, centre, DIODE_OUTER_RADIUS_CM),
      255
  );
  fill_poly(
      mask,
      board_circle_image(image_to_board, centre, DIODE_INNER_RADIUS_CM),
      0
  );
  return mask;
}

/**
 * Find the diode in a colour view by its redness.
 *
 * Redness is the red channel minus the brighter of green and blue, which
 * markers, board and floor never show.
 *
 * @param[in] bgr             colour image of the search region
 * @param[in] mask            pixels the diode may occupy
 * @param[in] image_to_board  pose used to reject candidates off the orbit
 * @param[in] offset          position of the region within the frame
 * @returns the diode in frame coordinates, or nothing if no candidate is red
 *          enough or all candidates lie outside the orbit band
 * @exceptsafe basic
 */
std::optional<cv::Point2d> find_red_led(
    const cv::Mat& bgr,
    const cv::Mat& mask,
    const cv::Mat& image_to_board,
    const cv::Point2d& orbit_centre,
    const cv::Point2d& offset = cv::Point2d(
        0.0,
        0.0
    )
) {
  std::vector<cv::Mat> channels;
  cv::Mat signed_bgr;
  bgr.convertTo(signed_bgr, CV_16SC3);
  cv::split(signed_bgr, channels);
  cv::Mat brighter_of_gb = cv::max(channels[1], channels[0]);
  cv::Mat redness_signed = channels[2] - brighter_of_gb;
  cv::Mat redness;
  redness_signed.convertTo(redness, CV_8UC1);
  cv::Mat masked_redness = cv::Mat::zeros(redness.size(), CV_8UC1);
  redness.copyTo(masked_redness, mask);
  redness = masked_redness;
  cv::GaussianBlur(redness, redness, cv::Size(0, 0), 2);

  double peak_value = 0.0;
  cv::minMaxLoc(redness, nullptr, &peak_value);
  const int peak = static_cast<int>(peak_value);
  if (peak < RED_LED_MIN_SCORE) return std::nullopt;

  const int floor_value = peak - 10;
  double weight_sum = 0.0, x_sum = 0.0, y_sum = 0.0;
  for (int y = 0; y < redness.rows; ++y) {
    const uchar* row = redness.ptr<uchar>(y);
    for (int x = 0; x < redness.cols; ++x) {
      if (row[x] < floor_value) continue;
      const double w = row[x];
      weight_sum += w;
      x_sum += x * w;
      y_sum += y * w;
    }
  }
  if (weight_sum <= 0.0) return std::nullopt;
  const cv::Point2d centre(
      x_sum / weight_sum + offset.x,
      y_sum / weight_sum + offset.y
  );
  if (!image_to_board.empty() &&
      !radius_in_band(image_to_board, orbit_centre, centre))
    return std::nullopt;
  return centre;
}

/**
 * Report whether a view carries no colour.
 *
 * The robot cameras stream infrared, where redness cannot find the diode and
 * brightness has to be used instead.
 *
 * @param[in] bgr   colour image of the search region
 * @param[in] mask  pixels to measure
 * @returns true when mean chroma is below the greyscale limit
 * @exceptsafe basic
 */
bool is_gray_view(
    const cv::Mat& bgr,
    const cv::Mat& mask
) {
  std::vector<cv::Mat> channels;
  cv::split(bgr, channels);
  cv::Mat red_green, green_blue;
  cv::absdiff(channels[2], channels[1], red_green);
  cv::absdiff(channels[1], channels[0], green_blue);
  const double chroma =
      cv::mean(red_green, mask)[0] + cv::mean(green_blue, mask)[0];
  return chroma < GRAY_CHROMA_LIMIT;
}

/**
 * Find the diode in a greyscale view by its brightness.
 *
 * Successively lower thresholds are tried, and the brightest small blob
 * inside the orbit band wins.
 *
 * @param[in] gray            greyscale image of the search region
 * @param[in] mask            pixels the diode may occupy
 * @param[in] image_to_board  pose used to reject candidates off the orbit
 * @param[in] offset          position of the region within the frame
 * @returns the diode in frame coordinates, or nothing if no blob qualifies
 * @exceptsafe basic
 */
std::optional<cv::Point2d> find_bright_diode(
    const cv::Mat& gray,
    const cv::Mat& mask,
    const cv::Mat& image_to_board,
    const cv::Point2d& orbit_centre,
    const cv::Point2d& offset = cv::Point2d(
        0.0,
        0.0
    )
) {
  if (image_to_board.empty()) return std::nullopt;
  cv::Mat masked = cv::Mat::zeros(gray.size(), CV_8UC1);
  gray.copyTo(masked, mask);
  for (int threshold : BRIGHT_THRESHOLDS) {
    cv::Mat binary;
    cv::threshold(masked, binary, threshold, 255, cv::THRESH_BINARY);
    cv::Mat labels, stats, centroids;
    const int count =
        cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    std::vector<int> peak(count, 0);
    for (int y = 0; y < masked.rows; ++y) {
      const int* labelled = labels.ptr<int>(y);
      const uchar* row = masked.ptr<uchar>(y);
      for (int x = 0; x < masked.cols; ++x)
        if (labelled[x] > 0 && row[x] > peak[labelled[x]])
          peak[labelled[x]] = row[x];
    }

    std::optional<cv::Point2d> brightest;
    int best_peak = 0;
    for (int label = 1; label < count; ++label) {
      const int area = stats.at<int>(label, cv::CC_STAT_AREA);
      if (area < BRIGHT_MIN_AREA || area > BRIGHT_MAX_AREA) continue;
      const cv::Point2d centre(
          centroids.at<double>(label, 0) + offset.x,
          centroids.at<double>(label, 1) + offset.y
      );
      if (!radius_in_band(image_to_board, orbit_centre, centre)) continue;
      if (peak[label] > best_peak) {
        best_peak = peak[label];
        brightest = centre;
      }
    }
    if (brightest) return brightest;
  }
  return std::nullopt;
}

/**
 * The diode's track so far, projected into a panel.
 *
 * Frames where the LED was not found hold non-finite positions and are
 * dropped, so the drawn trail shows only real detections.
 *
 * @param[in] analysis  run state holding the track
 * @param[in] view      which view
 * @param[in] to_board  panel-to-board homography
 * @returns the trail in panel pixels, empty if nothing was ever found
 * @exceptsafe basic
 */
std::vector<cv::Point2f> diode_trail(
    const Analysis& analysis,
    int view,
    const cv::Mat& to_board
) {
  std::vector<cv::Point2f> finite;
  for (const auto& p : analysis.diodes[view].positions_cm)
    if (is_finite(p)) finite.emplace_back(p.x, p.y);
  if (finite.empty()) return {};
  return board_to_image_points(to_board, finite);
}

/**
 * The fitted orbit circle, projected into a panel.
 *
 * @param[in] analysis  run state holding the fitted circle
 * @param[in] view      which view
 * @param[in] to_board  panel-to-board homography
 * @returns the circle as a panel-space polyline
 * @exceptsafe basic
 */
std::vector<cv::Point2f> orbit_trail(
    const Analysis& analysis,
    int view,
    const cv::Mat& to_board
) {
  const Circle& circle = analysis.circles[view];
  return board_to_image_points(
      to_board,
      orbit_points(circle.center, circle.radius)
  );
}

/**
 * Whether a view resolved a pose in this frame.
 *
 * @param[in] view  the view's geometry
 * @returns true when its homography was fitted
 * @exceptsafe no-throw
 */
bool fitted(const ViewGeometry& view) { return !view.image_to_board.empty(); }

/**
 * This frame's diode detection, in panel pixels.
 *
 * @param[in] analysis  run state
 * @param[in] view      which view
 * @param[in] to_panel  frame-to-panel transform
 * @returns the detection mapped into the panel
 * @exceptsafe basic
 */
cv::Point2d diode_in_panel(
    const Analysis& analysis,
    int view,
    const cv::Matx33d& to_panel
) {
  const cv::Point2d& pixel = analysis.diodes[view].pixels[analysis.index];
  return apply_transform(to_panel, pixel);
}

/**
 * Per-view transforms for drawing into a panel.
 *
 * Composes the frame-to-panel crop with each view's own homography, so board
 * geometry can be drawn straight into panel coordinates. Views that did not
 * resolve are left unfitted and skipped by the drawing code.
 *
 * @param[in] analysis  run state holding this frame's geometry
 * @param[in] to_panel  frame-to-panel transform
 * @returns one entry per view
 * @exceptsafe basic
 */
PanelViews cropped_views(
    const Analysis& analysis,
    const cv::Matx33d& to_panel
) {
  PanelViews views(VIEW_COUNT);
  for (int view = 0; view < VIEW_COUNT; ++view) {
    const ViewGeometry& geom = analysis.geometry[view];
    if (!fitted(geom)) continue;
    views[view].fitted = true;
    views[view].to_panel = to_panel;
    views[view].to_board = geom.image_to_board * cv::Mat(to_panel.inv());
  }
  return views;
}

/**
 * Where a view's annulus is drawn in the band panel.
 *
 * The passthrough gets the upper half to itself and the robot cameras share
 * the lower, spaced evenly however many there are. Fixed slots rather than
 * wherever the board happens to be, so the views can be compared directly.
 *
 * @param[in] view  which view
 * @returns the slot's centre in panel pixels
 * @exceptsafe basic
 */
cv::Point2d band_slot_centre(int view) {
  if (view == TOP_VIEW)
    return cv::Point2d(PANEL_SIZE * 0.5, PANEL_SIZE * BAND_TOP_CENTRE_Y);
  const auto at = std::find(BOTTOM_VIEWS.begin(), BOTTOM_VIEWS.end(), view);
  const size_t slot = at - BOTTOM_VIEWS.begin();
  const double across = (2.0 * slot + 1.0) / (2.0 * BOTTOM_VIEWS.size());
  return cv::Point2d(PANEL_SIZE * across, PANEL_SIZE * BAND_BOTTOM_CENTRE_Y);
}

/**
 * Scale that fits a view's annulus into its band slot.
 *
 * @param[in] view  which view
 * @returns panel pixels per board centimetre for that slot
 * @exceptsafe no-throw
 */
double band_slot_scale(int view) {
  const double diameter =
      view == TOP_VIEW ? BAND_TOP_DIAMETER : BAND_BOTTOM_DIAMETER;
  return PANEL_SIZE * diameter / (2.0 * DIODE_OUTER_RADIUS_CM);
}

/**
 * Look straight down on every board, north up, in a slot the panel fixes.
 *
 * Swaps a view's perspective for a plain scale about its orbit centre, so the
 * board plane is seen face on: the band reads as a true circle instead of an
 * ellipse, and the board's own up — the direction the printed markers ascend,
 * which is north — points up the panel however the board lay in the frame.
 * The slot places and sizes each band outright, so the bands hold still
 * instead of drifting with wherever the cameras happened to frame them.
 *
 * @param[in] analysis  geometry, tracks and fits for this frame
 * @returns one rectified transform per view, unfitted views left empty
 * @exceptsafe basic
 */
PanelViews rectified_views(const Analysis& analysis) {
  PanelViews views(VIEW_COUNT);
  for (int view = 0; view < VIEW_COUNT; ++view) {
    const ViewGeometry& geom = analysis.geometry[view];
    if (!fitted(geom)) continue;
    const cv::Point2d& centre = analysis.circles[view].center;
    const cv::Point2d at = band_slot_centre(view);
    const double scale = band_slot_scale(view);
    const cv::Matx33d board_to_panel(
        scale,
        0.0,
        at.x - scale * centre.x,
        0.0,
        scale,
        at.y - scale * centre.y,
        0.0,
        0.0,
        1.0
    );
    const cv::Mat image_to_panel =
        cv::Mat(board_to_panel) * geom.image_to_board;
    views[view].fitted = true;
    views[view].to_panel = image_to_panel;
    views[view].to_board = cv::Mat(board_to_panel.inv());
  }
  return views;
}

/**
 * Add one diode position to the running circle fit.
 *
 * The fit gives the orbit centre that every angle is measured against.
 *
 * @param[in,out] state  accumulated normal equations and current circle
 * @param[in]     point  diode position in board coordinates
 * @exceptsafe basic
 */
void circle_add(
    RunningCircle& state,
    const cv::Point2d& point
) {
  const cv::Vec3d row(2.0 * point.x, 2.0 * point.y, 1.0);
  const double target = point.x * point.x + point.y * point.y;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) state.ata(r, c) += row[r] * row[c];
    state.atb[r] += row[r] * target;
  }
  ++state.count;
  if (state.count < 3) return;
  cv::Mat a(3, 3, CV_64F), b(3, 1, CV_64F), solution;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) a.at<double>(r, c) = state.ata(r, c);
    b.at<double>(r, 0) = state.atb[r];
  }
  if (!cv::solve(a, b, solution, cv::DECOMP_SVD)) return;
  const double cx = solution.at<double>(0), cy = solution.at<double>(1);
  const double squared = solution.at<double>(2) + cx * cx + cy * cy;
  if (squared <= 0.0) return;
  state.circle = Circle{cv::Point2d(cx, cy), std::sqrt(squared)};
}

/**
 * Add one angle sample to the running fit of angle against time.
 *
 * @param[in,out] state  accumulated sums for this view
 * @param[in]     x      time of the sample in seconds
 * @param[in]     y      unwrapped angle in radians
 * @exceptsafe no-throw
 */
void regression_add(
    RunningRegression& state,
    double x,
    double y
) {
  state.n += 1.0;
  state.sx += x;
  state.sxx += x * x;
  state.sy += y;
  state.sxy += x * y;
  state.syy += y * y;
}

/**
 * Turn a wrapped angle into a continuous one.
 *
 * The rotation direction is locked once, from the reference view, and applied
 * as the same multiplicative flip to every view, so it cancels in the phase
 * difference the latency is derived from.
 *
 * @param[in,out] state              per-view unwrapping state
 * @param[in]     raw                wrapped angle from this frame
 * @param[in,out] rotation           direction shared by all views
 * @param[in]     is_reference_view  whether this view may lock the direction
 * @returns the continuous angle in radians
 * @exceptsafe no-throw
 */
double unwrap_add(
    RunningUnwrap& state,
    double raw,
    RotationSign& rotation,
    bool is_reference_view
) {
  if (!state.has_previous) {
    state.has_previous = true;
    state.previous_raw = raw;
    return rotation.value * raw;
  }
  const double delta = raw - state.previous_raw;
  double wrapped = std::fmod(delta + CV_PI, 2.0 * CV_PI);
  if (wrapped < 0.0) wrapped += 2.0 * CV_PI;
  wrapped -= CV_PI;
  if (wrapped == -CV_PI && delta > 0.0) wrapped = CV_PI;
  if (is_reference_view) lock_rotation_sign(rotation, wrapped);
  if (std::abs(delta) >= CV_PI) state.correction += wrapped - delta;
  state.previous_raw = raw;
  return rotation.value * (raw + state.correction);
}

/**
 * Derive the stream latency from the per-view fits.
 *
 * Every view watches the same diode, so their fitted lines share a slope and
 * differ only in phase; dividing that phase gap by the shared angular speed
 * converts it into time. Fitting over all frames is what averages the
 * per-frame noise out of the gap.
 *
 * @param[in] analysis  running fits and the paired-frame time span
 * @returns the per-view fits, mean angular speed and latency in milliseconds
 * @exceptsafe basic
 */
LatencyResult latency_from(const Analysis& analysis) {
  LatencyResult result;
  result.fits.resize(VIEW_COUNT);
  result.paired_frames = analysis.paired;
  double omega_sum = 0.0;
  for (int view = 0; view < VIEW_COUNT; ++view) {
    result.fits[view] = regression_fit(analysis.regressions[view]);
    omega_sum += result.fits[view].slope;
  }
  result.omega = omega_sum / VIEW_COUNT;

  double latency_sum = 0.0;
  const RegressionFit& top = result.fits[TOP_VIEW];
  for (int view : BOTTOM_VIEWS) {
    const double slope_gap = top.slope - result.fits[view].slope;
    const double intercept_gap = top.intercept - result.fits[view].intercept;
    const double start = wrap(slope_gap * analysis.first_time + intercept_gap);
    const double end = wrap(slope_gap * analysis.last_time + intercept_gap);
    const double average_gap = 0.5 * (start + end);
    latency_sum +=
        result.omega != 0.0 ? average_gap / result.omega * 1000.0 : NaN;
  }
  result.latency_ms = latency_sum / static_cast<double>(BOTTOM_VIEWS.size());
  return result;
}

/**
 * Fix the view count and labels once the layout is known.
 *
 * Called after the opening frames reveal whether the robot camera is mono or
 * a stereo pair, which is why the view globals are not constants: the same
 * binary handles both without a flag.
 *
 * @param[in] two_bottom_boards  true when the robot camera is stereo
 * @exceptsafe basic
 */
void configure_views(bool two_bottom_boards) {
  TWO_BOTTOM_BOARDS = two_bottom_boards;
  VIEW_NAMES = {"headset passthrough", "robot camera"};
  BOTTOM_VIEWS = {1};
  if (two_bottom_boards) {
    VIEW_NAMES = {"headset passthrough", "robot camera L", "robot camera R"};
    BOTTOM_VIEWS = {1, 2};
  }
  VIEW_COUNT = static_cast<int>(VIEW_NAMES.size());
}

/**
 * Build the detector, board model and contrast equaliser together.
 *
 * CLAHE is part of the detector because the infrared preview is too flat to
 * decode raw; equalising it locally is what lets one parameter set serve both
 * views. Corner refinement is enabled here, since the homography's accuracy —
 * and so the latency — rests on sub-pixel corners.
 *
 * @returns the assembled detector
 * @exceptsafe basic
 */
Detector make_detector() {
  const cv::aruco::RefineParameters refine(12.0f, 3.0f, true);
  return Detector{
      cv::aruco::ArucoDetector(
          make_dictionary(),
          make_detector_params(),
          refine
      ),
      make_board(),
      cv::createCLAHE(3.0, cv::Size(8, 8))
  };
}

/**
 * Count the boards visible in one frame.
 *
 * @param[in]     bgr       frame to inspect
 * @param[in,out] detector  dictionary and tuned detector parameters
 * @returns the highest number of times any single marker id appears, which is
 *          how many boards the frame shows
 * @exceptsafe basic
 */
int count_boards_in_frame(
    const cv::Mat& bgr,
    Detector& detector
) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  return boards_in_frame(detect_valid_markers(gray, detector));
}

/**
 * Open a clip, or fail loudly.
 *
 * @param[in] input_path  file to open
 * @returns the opened capture
 * @throws std::runtime_error if it cannot be opened
 * @exceptsafe basic
 */
cv::VideoCapture open_video(const std::string& input_path) {
  cv::VideoCapture capture(input_path);
  if (!capture.isOpened())
    throw std::runtime_error("cannot open " + input_path);
  return capture;
}

/**
 * Average a board pose over time.
 *
 * The boards are static in the room and only the headset moves, so an
 * exponential average of the four projected board corners removes fit noise
 * that is otherwise very visible on the small robot camera views. Weights are
 * normalised so early frames are not biased by the first sample.
 *
 * @param[in,out] state           running weighted corners for this view
 * @param[in]     image_to_board  pose fitted from this frame alone
 * @returns the smoothed image-to-board homography
 * @exceptsafe basic
 */
cv::Mat smooth_pose(
    PoseSmoother& state,
    const cv::Mat& image_to_board
) {
  const std::vector<cv::Point2f> now = project_board_outline(image_to_board);
  const float decay = 1.0f - POSE_SMOOTHING;
  if (state.weighted.size() != now.size())
    state.weighted.assign(now.size(), cv::Point2f(0.0f, 0.0f));
  state.weight = 1.0 + decay * state.weight;

  std::vector<cv::Point2f> smoothed(now.size());
  for (size_t i = 0; i < now.size(); ++i) {
    state.weighted[i] = now[i] + decay * state.weighted[i];
    smoothed[i] = state.weighted[i] / static_cast<float>(state.weight);
  }

  const float hw = HALF_WIDTH_CM, hh = HALF_HEIGHT_CM;
  const std::vector<cv::Point2f> outline{
      {-hw, -hh},
      {hw, -hh},
      {hw, hh},
      {-hw, hh}
  };
  const cv::Mat board_to_image = cv::findHomography(outline, smoothed);
  if (board_to_image.empty()) return image_to_board;
  return board_to_image.inv();
}

/**
 * Find and fit every board in one frame.
 *
 * A board cannot move far between frames, so each view is searched in a grown
 * box around where it was last seen; if any view is lost the frame falls back
 * to a full-frame rescan.
 *
 * @param[in]     bgr       frame to analyse
 * @param[in,out] detector  dictionary, board model and detector parameters
 * @param[in,out] tracked   bounds carried between frames, updated in place
 * @returns the fitted geometry of every view; a view that could not be fitted
 *          comes back with an empty homography
 * @exceptsafe basic
 */
FrameGeometry detect_view_geometry(
    const cv::Mat& bgr,
    Detector& detector,
    std::vector<CropBox>& tracked
) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  FrameGeometry geometry(VIEW_COUNT);

  bool have_track = static_cast<int>(tracked.size()) == VIEW_COUNT;
  for (const CropBox& box : tracked)
    if (!box_is_usable(box)) have_track = false;

  if (have_track) {
    for (int view = 0; view < VIEW_COUNT && have_track; ++view) {
      const CropBox search = grow_box(tracked[view], gray.size());
      MarkerHits hits = detect_markers_in_crop(gray, search, detector);
      keep_nearest_to_view(hits, tracked, view);
      geometry[view] = fit_view(std::move(hits));
      have_track = fitted(geometry[view]);
    }
  }

  if (!have_track) {
    const ViewBoxes scan = view_crop_boxes(gray, detector);
    for (int view = 0; view < VIEW_COUNT; ++view) {
      const CropBox& bounds = scan.bounds[view];
      const CropBox search = pad_box(bounds, CROP_PADDING_PX, gray.size());
      MarkerHits hits = detect_markers_in_crop(gray, search, detector);
      keep_nearest_to_view(hits, scan.bounds, view);
      geometry[view] = fit_view(std::move(hits));
    }
  }

  tracked.assign(VIEW_COUNT, CropBox{0, 0, 0, 0});
  for (int view = 0; view < VIEW_COUNT; ++view)
    if (fitted(geometry[view]))
      tracked[view] = corners_bounding_box(geometry[view].corners);
  return geometry;
}

/**
 * Square crop holding every board plus a margin.
 *
 * Keeps the panels showing the boards rather than the whole room.
 *
 * @param[in] geometry  fitted geometry of every view
 * @param[in] size      frame size the crop is clamped to
 * @returns the crop in frame coordinates
 * @exceptsafe basic
 */
cv::Rect board_crop_rect(
    const FrameGeometry& geometry,
    const cv::Size& size
) {
  std::vector<cv::Point2f> points;
  for (int view = 0; view < VIEW_COUNT; ++view)
    for (const auto& quad : geometry[view].corners)
      for (const auto& p : quad) points.push_back(p);
  if (points.empty()) return cv::Rect(0, 0, size.width, size.height);

  const Bounds b = bounds_of(points);
  int x0 = std::min(size.width, static_cast<int>(b.x0));
  int y0 = std::min(size.height, static_cast<int>(b.y0));
  int x1 = std::max(0, static_cast<int>(b.x1));
  int y1 = std::max(0, static_cast<int>(b.y1));
  const int margin = track_margin(std::max(x1 - x0, y1 - y0));
  x0 -= margin;
  y0 -= margin;
  x1 += margin;
  y1 += margin;

  int side = std::max(x1 - x0, y1 - y0);
  side = std::min(side, std::min(size.width, size.height));
  int x = (x0 + x1) / 2 - side / 2;
  int y = (y0 + y1) / 2 - side / 2;
  x = std::max(0, std::min(x, size.width - side));
  y = std::max(0, std::min(y, size.height - side));
  return cv::Rect(x, y, side, side);
}

/**
 * Average the panel crop over time.
 *
 * The per-frame crop jitters with every corner refinement, which reads as the
 * panel shaking. The smoothed square is grown when it would clip the boards,
 * because lag is acceptable and losing a board off the edge is not.
 *
 * @param[in,out] state   running centre and side of the crop
 * @param[in]     target  crop wanted for this frame
 * @param[in]     size    frame size the crop is clamped to
 * @param[in]     fps     frame rate, which sets the averaging time constant
 * @returns the smoothed crop in frame coordinates
 * @exceptsafe no-throw
 */
cv::Rect smooth_crop_rect(
    CropSmoother& state,
    const cv::Rect& target,
    const cv::Size& size,
    double fps
) {
  const double alpha = 1.0 - std::exp(-1.0 / (fps * CROP_SMOOTHING_SECONDS));
  const double target_cx = target.x + 0.5 * target.width;
  const double target_cy = target.y + 0.5 * target.height;
  const double target_side = target.width;

  if (!state.initialised) {
    state.cx = target_cx;
    state.cy = target_cy;
    state.side = target_side;
    state.initialised = true;
  } else {
    state.cx += alpha * (target_cx - state.cx);
    state.cy += alpha * (target_cy - state.cy);
    state.side += alpha * (target_side - state.side);
  }

  double side = state.side * (1.0 + CROP_SLACK);
  double left = state.cx - 0.5 * side;
  double top = state.cy - 0.5 * side;
  double overshoot = std::max(left - target.x, top - target.y);
  overshoot = std::max(overshoot, target.x + target.width - left - side);
  overshoot = std::max(overshoot, target.y + target.height - top - side);
  if (overshoot > 0.0) {
    side += 2.0 * overshoot;
    left = state.cx - 0.5 * side;
    top = state.cy - 0.5 * side;
  }

  int square = static_cast<int>(std::lround(side));
  square = std::min(square, std::min(size.width, size.height));
  int x = static_cast<int>(std::lround(left));
  int y = static_cast<int>(std::lround(top));
  x = std::max(0, std::min(x, size.width - square));
  y = std::max(0, std::min(y, size.height - square));
  return cv::Rect(x, y, square, square);
}

/**
 * Transform mapping a crop of the frame onto a panel.
 *
 * Uniform scale from the crop's width, so the aspect is preserved and the
 * board is not stretched; the translation places the crop's origin.
 *
 * @param[in] crop  region of the frame the panel shows
 * @returns the frame-to-panel homography
 * @exceptsafe no-throw
 */
cv::Matx33d panel_transform(const cv::Rect& crop) {
  const double scale = static_cast<double>(PANEL_SIZE) / crop.width;
  return cv::Matx33d(
      scale,
      0.0,
      -scale * crop.x,
      0.0,
      scale,
      -scale * crop.y,
      0.0,
      0.0,
      1.0
  );
}

/**
 * Lay a compass rose under every fitted board of a panel.
 *
 * The star is drawn onto a copy and blended back at `COMPASS_OPACITY`; the
 * letters go on afterwards at full strength. Called onto the bare panel
 * before the footage is pasted over it, so the rose ends up behind the boards
 * as well as behind every marker, orbit and trail, and survives only where
 * the panel shows background.
 *
 * @param[in,out] canvas    panel drawn into
 * @param[in]     analysis  geometry, tracks and fits for this frame
 * @exceptsafe basic
 */
void render_compasses(
    cv::Mat& canvas,
    const Analysis& analysis,
    const PanelViews& views
) {
  cv::Mat rose = canvas.clone();
  for (int view = 0; view < VIEW_COUNT; ++view) {
    if (!views[view].fitted) continue;
    draw_compass_rose(
        rose,
        views[view].to_board,
        analysis.circles[view].center
    );
  }
  cv::addWeighted(
      rose,
      COMPASS_OPACITY,
      canvas,
      1.0 - COMPASS_OPACITY,
      0.0,
      canvas
  );

  for (int view = 0; view < VIEW_COUNT; ++view) {
    if (!views[view].fitted) continue;
    draw_compass_labels(
        canvas,
        views[view].to_board,
        analysis.circles[view].center
    );
  }
}

/**
 * Draw the detected geometry over the cleaned frame.
 *
 * Shows markers found and markers merely projected, the board outline, the
 * fitted orbit and the diode.
 *
 * @param[in,out] canvas    panel drawn into
 * @param[in]     analysis  geometry, tracks and fits for this frame
 * @exceptsafe basic
 */
void render_overlay(
    cv::Mat& canvas,
    const Analysis& analysis,
    const PanelViews& views
) {
  for (int view = 0; view < VIEW_COUNT; ++view) {
    const ViewGeometry& geom = analysis.geometry[view];
    if (!views[view].fitted) continue;
    const cv::Mat& to_board = views[view].to_board;
    const std::set<int> found(geom.ids.begin(), geom.ids.end());
    for (const auto& entry : project_all_markers(to_board)) {
      const cv::Scalar color = found.count(entry.first) ? GREEN : ORANGE;
      draw_poly(canvas, entry.second, true, color, 2);
    }
    draw_poly(canvas, project_board_outline(to_board), true, BLUE, 3);

    const cv::Point2d& center = analysis.circles[view].center;
    const cv::Point2f origin(center.x, center.y);
    const cv::Point2f pivot_top = board_to_image_points(to_board, {origin})[0];
    const double pixels_per_cm = board_pixels_per_cm(to_board, center);
    const cv::Point2f floor_drop(0.0f, DIODE_HEIGHT_CM * pixels_per_cm);

    std::vector<cv::Point2f> trail = diode_trail(analysis, view, to_board);
    for (auto& p : trail) p += floor_drop;
    if (!trail.empty()) draw_poly(canvas, trail, false, YELLOW, 1);

    std::vector<cv::Point2f> orbit = orbit_trail(analysis, view, to_board);
    for (auto& p : orbit) p += floor_drop;
    draw_poly(canvas, orbit, true, MAGENTA, 1);

    const cv::Point2f pivot_base = pivot_top + floor_drop;
    const cv::Point2d spot =
        diode_in_panel(analysis, view, views[view].to_panel);
    draw_line(canvas, pivot_base, pivot_top, WHITE);
    if (is_finite(spot)) {
      const cv::Point2f tip(spot.x, spot.y);
      draw_line(canvas, pivot_top, tip, WHITE);
    }
  }
}

/**
 * Draw the orbit band alone, with each view's diode circled.
 *
 * The diode pixels are snapshotted before the overlay is drawn, so the orbit
 * and trail lines cannot paint over the diode itself.
 *
 * @param[in,out] canvas    panel drawn into
 * @param[in]     analysis  geometry, tracks and fits for this frame
 * @exceptsafe basic
 */
void render_band(
    cv::Mat& canvas,
    const Analysis& analysis,
    const PanelViews& views
) {
  const int marker_radius = 28;
  const cv::Rect canvas_rect(0, 0, canvas.cols, canvas.rows);
  std::vector<cv::Point> centres(VIEW_COUNT, cv::Point(-1, -1));
  std::vector<cv::Point2f> spots(VIEW_COUNT);
  std::vector<cv::Rect> boxes(VIEW_COUNT);
  std::vector<cv::Mat> pristine(VIEW_COUNT);
  for (int view = 0; view < VIEW_COUNT; ++view) {
    const cv::Point2d spot =
        diode_in_panel(analysis, view, views[view].to_panel);
    if (!is_finite(spot)) continue;
    const cv::Point centre(cvRound(spot.x), cvRound(spot.y));
    const cv::Point corner(centre.x - marker_radius, centre.y - marker_radius);
    const int span = 2 * marker_radius + 1;
    const cv::Rect box = cv::Rect(corner.x, corner.y, span, span) & canvas_rect;
    if (box.width <= 0 || box.height <= 0) continue;
    centres[view] = centre;
    spots[view] = cv::Point2f(spot.x, spot.y);
    boxes[view] = box;
    pristine[view] = canvas(box).clone();
  }

  for (int view = 0; view < VIEW_COUNT; ++view) {
    if (!views[view].fitted) continue;
    const cv::Mat& to_board = views[view].to_board;
    const std::vector<cv::Point2f> trail =
        diode_trail(analysis, view, to_board);
    if (!trail.empty()) draw_poly(canvas, trail, false, YELLOW, 1);
    draw_poly(canvas, orbit_trail(analysis, view, to_board), true, MAGENTA, 2);
  }

  for (int view = 0; view < VIEW_COUNT; ++view) {
    if (centres[view].x < 0) continue;
    cv::Mat keep = cv::Mat::zeros(boxes[view].size(), CV_8UC1);
    const cv::Point local = centres[view] - boxes[view].tl();
    cv::circle(keep, local, marker_radius - 1, cv::Scalar(255), cv::FILLED);
    cv::Mat target = canvas(boxes[view]);
    pristine[view].copyTo(target, keep);
    draw_circle(canvas, spots[view], marker_radius, VIEW_COLORS[view], 2);
  }
}

/**
 * Join strings with a separator, for the one-line summaries.
 *
 * @param[in] parts    strings to join
 * @param[in] between  separator placed between them
 * @returns the joined string
 * @exceptsafe basic
 */
std::string join_views(
    const std::vector<std::string>& parts,
    const char* between
) {
  std::string joined;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) joined += between;
    joined += parts[i];
  }
  return joined;
}

/**
 * Write out how the latency was reached.
 *
 * States it in the plain terms a viewer of the video needs, mixing the values
 * of this frame with those fitted over the whole clip.
 *
 * @param[in,out] cursor    text cursor, advanced line by line
 * @param[in]     analysis  geometry, tracks and fits for this frame
 * @exceptsafe basic
 */
void explain_latency(
    TextCursor& cursor,
    const Analysis& analysis
) {
  const LatencyResult& result = analysis.result;
  const double omega = std::abs(result.omega);
  const double speed = revolutions(omega);
  const double gap = revolutions(result.latency_ms / 1000.0 * omega);

  std::vector<std::string> angles, speeds, others;
  double worst_fit = 1.0;
  for (int view = 0; view < VIEW_COUNT; ++view) {
    const RegressionFit& fit = result.fits[view];
    const double turned = revolutions(analysis.angles[view][analysis.index]);
    angles.push_back(VIEW_COLOR_NAMES[view] + (" " + number(turned, 2)));
    speeds.push_back(number(revolutions(fit.slope), 3));
    worst_fit = std::min(worst_fit, fit.r2);
    if (view != TOP_VIEW) others.push_back(VIEW_COLOR_NAMES[view]);
  }
  const std::string rest = join_views(others, "+");

  print_line(cursor, "A single diode spins on the board at a steady rate.");
  print_line(
      cursor,
      "%d views track it, marked by the circles below:",
      VIEW_COUNT
  );
  print_line(
      cursor,
      "%s = headset passthrough (what you see live),",
      VIEW_COLOR_NAMES[TOP_VIEW]
  );
  print_line(cursor, "%s = the robot cameras, streamed back.", rest.c_str());
  print_line(cursor, "This frame: %s rev.", join_views(angles, ", ").c_str());
  print_line(
      cursor,
      "Across %d frames every view spins at one rate:",
      result.paired_frames
  );
  print_line(
      cursor,
      "%s rev/s, all fitting R2 %s+.",
      join_views(speeds, ", ").c_str(),
      number(worst_fit, 3).c_str()
  );
  print_line(
      cursor,
      "But %s sits %+.4f rev behind %s.",
      rest.c_str(),
      gap,
      VIEW_COLOR_NAMES[TOP_VIEW]
  );
  print_line(cursor, "At %.4f rev/s that angle is a time offset,", speed);
  print_line(
      cursor,
      "so %+.4f / %.4f = %+.1f ms,",
      gap,
      speed,
      result.latency_ms
  );
  print_line(cursor, "measured as camera - passthrough.");
}

/**
 * Draw the measurement table and the latency panel.
 *
 * @param[in,out] panel     panel drawn into
 * @param[in]     analysis  geometry, tracks and fits for this frame
 * @exceptsafe basic
 */
void render_numbers(
    cv::Mat& panel,
    const Analysis& analysis
) {
  const LatencyResult& result = analysis.result;
  const int index = analysis.index;
  const int paired = result.paired_frames;
  const double seconds = index / analysis.fps;
  panel.setTo(SITE_GREEN);

  TextCursor cursor{&panel, TEXT_MARGIN, TEXT_MARGIN, TEXT_HEIGHT, TEXT_LINE};
  print_line(cursor, "frame %d/%d    t = %.3f s", index, paired, seconds);
  cursor.y += TEXT_LINE;

  const int markers = static_cast<int>(marker_ids().size());
  const int top = cursor.y;
  const int height = TEXT_HEIGHT * 7 / 9;
  const int line = height * 3 / 2;
  const std::string ruler(20, '0');
  const double step = text_width(ruler, height) / 20.0;
  const std::array<int, TABLE_COLUMNS> widths{22, 8, 7, 7, 7, 7, 7, 7, 7};
  const std::array<const char*, TABLE_COLUMNS> heads{
      "view",
      "markers",
      "X cm",
      "Y cm",
      "r cm",
      "rev",
      "rev/s",
      "R2",
      "RMSE"
  };

  std::array<int, TABLE_COLUMNS + 1> edge;
  edge[0] = TEXT_MARGIN;
  for (int c = 0; c < TABLE_COLUMNS; ++c)
    edge[c + 1] = edge[c] + cvRound(widths[c] * step);
  const int text_top = (line - height) / 2;

  for (int c = 0; c < TABLE_COLUMNS; ++c) {
    const int slack = edge[c + 1] - edge[c] - text_width(heads[c], height);
    draw_text(panel, heads[c], edge[c] + slack / 2, top + text_top, height);
  }

  for (int view = 0; view < VIEW_COUNT; ++view) {
    const cv::Point2d spot = analysis.diodes[view].positions_cm[index];
    const double angle = revolutions(analysis.angles[view][index]);
    const double radius = analysis.circles[view].radius;
    const int found = static_cast<int>(analysis.geometry[view].ids.size());
    const RegressionFit& fit = result.fits[view];
    char counted[16];
    std::snprintf(counted, sizeof(counted), "%d/%d", found, markers);
    const std::array<std::string, TABLE_COLUMNS> cells{
        VIEW_NAMES[view],
        counted,
        number(spot.x, 2),
        number(spot.y, 2),
        number(radius, 2),
        number(angle, 2),
        number(revolutions(fit.slope), 3),
        number(fit.r2, 4),
        number(revolutions(fit.rmse), 4)
    };
    const int y = top + (view + 1) * line + text_top;
    for (int c = 0; c < TABLE_COLUMNS; ++c) {
      const int slack = edge[c + 1] - edge[c] - text_width(cells[c], height);
      draw_text(panel, cells[c], edge[c] + slack / 2, y, height);
    }
  }

  const int rows = VIEW_COUNT + 1;
  for (int r = 0; r <= rows; ++r) {
    const cv::Point left(edge[0], top + r * line);
    cv::line(panel, left, cv::Point(edge[TABLE_COLUMNS], left.y), GRID, 1);
  }
  for (int c = 0; c <= TABLE_COLUMNS; ++c) {
    const cv::Point above(edge[c], top);
    cv::line(panel, above, cv::Point(above.x, top + rows * line), GRID, 1);
  }
  cursor.y = top + rows * line + TEXT_LINE;

  explain_latency(cursor, analysis);
  cursor.y += TEXT_LINE;

  cursor.height = TEXT_HEIGHT * 4;
  print_line(cursor, "%+.1f ms", result.latency_ms);
  draw_logo(
      panel,
      panel.cols - TEXT_MARGIN,
      panel.rows - TEXT_MARGIN,
      LOGO_HEIGHT
  );
}

/**
 * Mask the boards of every fitted view.
 *
 * The mask comes from the fitted homography rather than the detected markers,
 * so a frame that misses a few markers still keeps the whole board.
 *
 * @param[in] geometry  fitted geometry of every view
 * @param[in] size      frame size the mask is built for
 * @returns an 8-bit mask, set inside the boards
 * @exceptsafe basic
 */
cv::Mat board_mask_for(
    const FrameGeometry& geometry,
    const cv::Size& size
) {
  cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
  for (int view = 0; view < VIEW_COUNT; ++view) {
    const ViewGeometry& geom = geometry[view];
    if (!fitted(geom)) continue;
    fill_poly(mask, project_board_outline(geom.image_to_board), 255);
  }
  return mask;
}

/**
 * Allocate the run state, one slot per view.
 *
 * Every per-view vector is sized here and then only written by index, so no
 * later stage has to guard against a view that was never set up.
 *
 * @param[in] fps  the clip's frame rate, used to turn indices into seconds
 * @returns the initialised state
 * @exceptsafe basic
 */
Analysis make_analysis(double fps) {
  Analysis analysis;
  analysis.fps = fps;
  analysis.diodes.resize(VIEW_COUNT);
  analysis.circles.resize(VIEW_COUNT);
  analysis.angles.resize(VIEW_COUNT);
  analysis.running_circles.resize(VIEW_COUNT);
  analysis.regressions.resize(VIEW_COUNT);
  analysis.unwrappers.resize(VIEW_COUNT);
  analysis.result = latency_from(analysis);
  return analysis;
}

/**
 * Measure the diode in every view for one frame.
 *
 * A view without a fitted board, or a frame where any view loses the diode,
 * is recorded but left out of the latency fit.
 *
 * @param[in]     cleaned   frame with everything but the boards masked away
 * @param[in]     gray      greyscale version of `cleaned`
 * @param[in,out] analysis  tracks and running fits, extended by one frame
 * @exceptsafe basic
 */
void track_diodes(
    const cv::Mat& cleaned,
    const cv::Mat& gray,
    Analysis& analysis
) {
  const cv::Size size = cleaned.size();
  bool all_valid = true;
  for (int view = 0; view < VIEW_COUNT; ++view) {
    const ViewGeometry& geom = analysis.geometry[view];
    DiodeTrack& track = analysis.diodes[view];
    if (!fitted(geom)) {
      track.pixels.emplace_back(NaN, NaN);
      track.positions_cm.emplace_back(NaN, NaN);
      analysis.angles[view].push_back(NaN);
      all_valid = false;
      continue;
    }

    const cv::Mat& to_board = geom.image_to_board;
    const cv::Point2d& orbit = analysis.circles[view].center;
    const cv::Rect roi = board_annulus_bbox(to_board, orbit, size);
    std::optional<cv::Point2d> spot;
    if (roi.width > 1 && roi.height > 1) {
      cv::Mat board = cv::Mat::zeros(size, CV_8UC1);
      fill_poly(board, project_board_outline(to_board), 255);
      cv::Mat mask = board(roi);
      cv::bitwise_and(
          mask,
          board_annulus_mask(to_board, orbit, size)(roi),
          mask
      );
      const cv::Point2d offset(roi.x, roi.y);
      if (!is_gray_view(cleaned(roi), mask))
        spot = find_red_led(cleaned(roi), mask, to_board, orbit, offset);
      if (!spot)
        spot = find_bright_diode(gray(roi), mask, to_board, orbit, offset);
    }

    if (!spot) {
      track.pixels.emplace_back(NaN, NaN);
      track.positions_cm.emplace_back(NaN, NaN);
      analysis.angles[view].push_back(NaN);
      all_valid = false;
      continue;
    }
    const cv::Point2d board_point = image_to_board_point(to_board, *spot);
    track.pixels.push_back(*spot);
    track.positions_cm.push_back(board_point);
    circle_add(analysis.running_circles[view], board_point);
    analysis.circles[view] = analysis.running_circles[view].circle;
    const cv::Point2d& center = analysis.circles[view].center;
    const double raw =
        std::atan2(board_point.y - center.y, board_point.x - center.x);
    analysis.angles[view].push_back(unwrap_add(
        analysis.unwrappers[view],
        raw,
        analysis.rotation_sign,
        view == TOP_VIEW
    ));
  }

  if (!all_valid) return;
  const double seconds = analysis.index / analysis.fps;
  if (analysis.paired == 0) analysis.first_time = seconds;
  analysis.last_time = seconds;
  ++analysis.paired;
  for (int view = 0; view < VIEW_COUNT; ++view)
    regression_add(
        analysis.regressions[view],
        seconds,
        analysis.angles[view].back()
    );
  analysis.result = latency_from(analysis);
}

/**
 * Start the background encoder.
 *
 * Encoding a frame costs about as much as analysing one, so it runs on its
 * own thread rather than in the pipeline.
 *
 * @param[in,out] writer  encoder state to start
 * @param[in]     path    file to write
 * @param[in]     fps     frame rate of the output
 * @param[in]     size    frame size of the output
 * @throws std::runtime_error if the file cannot be opened for writing
 * @exceptsafe basic
 */
void writer_open(
    BackgroundWriter& writer,
    const std::string& path,
    double fps,
    const cv::Size& size
) {
  const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  writer.video.open(path, fourcc, fps, size);
  if (!writer.video.isOpened())
    throw std::runtime_error("cannot open video writer for " + path);
  writer.encoder = std::thread([&writer] { writer_encode_queued(writer); });
}

/**
 * Hand one frame to the background encoder.
 *
 * Frames cross over through a bounded pool of reusable buffers, so nothing is
 * allocated per frame and the pipeline only waits once the encoder falls
 * behind.
 *
 * @param[in,out] writer  encoder state
 * @param[in]     frame   frame to encode, copied before returning
 * @exceptsafe basic
 */
void writer_submit(
    BackgroundWriter& writer,
    const cv::Mat& frame
) {
  cv::Mat buffer;
  {
    std::unique_lock<std::mutex> lock(writer.mutex);
    writer.buffer_available.wait(lock, [&writer] {
      return !writer.spare.empty() || writer.allocated < WRITER_QUEUE_FRAMES;
    });
    if (writer.spare.empty()) {
      ++writer.allocated;
    } else {
      buffer = std::move(writer.spare.back());
      writer.spare.pop_back();
    }
  }
  frame.copyTo(buffer);
  {
    std::lock_guard<std::mutex> lock(writer.mutex);
    writer.queued.push_back(std::move(buffer));
  }
  writer.frame_queued.notify_one();
}

/**
 * Flush the encoder and shut it down.
 *
 * Sets the finishing flag under the lock and wakes the thread, so a frame
 * queued just before the last one is still written; joining before releasing
 * the video is what makes the output file complete.
 *
 * @param[in,out] writer  the writer to close
 * @exceptsafe basic
 */
void writer_close(BackgroundWriter& writer) {
  if (writer.encoder.joinable()) {
    {
      std::lock_guard<std::mutex> lock(writer.mutex);
      writer.finishing = true;
    }
    writer.frame_queued.notify_one();
    writer.encoder.join();
  }
  writer.video.release();
}

/**
 * Allocate the output grid and the four views onto it.
 *
 * The panels are OpenCV headers over one buffer rather than separate images,
 * so drawing into a panel writes straight into the frame that gets encoded
 * and no compositing step is needed. The numbers panel is drawn at 1920 and
 * scaled down, which is what keeps its text crisp.
 *
 * @returns the grid and its panel views
 * @exceptsafe basic
 */
Panels make_panels() {
  Panels panels;
  panels.stacked = cv::Mat(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3, SITE_GREEN);
  const int size = PANEL_SIZE;
  panels.raw = panels.stacked(cv::Rect(0, 0, size, size));
  panels.overlay = panels.stacked(cv::Rect(size, 0, size, size));
  panels.band = panels.stacked(cv::Rect(0, size, size, size));
  panels.numbers = panels.stacked(cv::Rect(size, size, size, size));
  panels.numbers_full = cv::Mat(1920, 1920, CV_8UC3);
  return panels;
}

/**
 * Draw the cross that separates the four panels.
 *
 * @param[in,out] stacked  the output grid
 * @exceptsafe basic
 */
void draw_panel_border(cv::Mat& stacked) {
  const cv::Point top(PANEL_SIZE, 0), bottom(PANEL_SIZE, OUTPUT_HEIGHT);
  const cv::Point left(0, PANEL_SIZE), right(OUTPUT_WIDTH, PANEL_SIZE);
  cv::line(stacked, top, bottom, WHITE, 2);
  cv::line(stacked, left, right, WHITE, 2);
}

/**
 * Scale an image into a panel.
 *
 * `INTER_AREA` because every use here is a downscale, where it avoids the
 * aliasing a bilinear resize leaves on the marker edges.
 *
 * @param[in]     source  image to scale
 * @param[in,out] panel   destination, its size fixed
 * @exceptsafe basic
 */
void fill_panel(
    const cv::Mat& source,
    cv::Mat& panel
) {
  cv::resize(source, panel, panel.size(), 0.0, 0.0, cv::INTER_AREA);
}

/**
 * Lay footage over whatever a panel already holds, masked to the boards.
 *
 * Leaves the panel's own background showing wherever the mask is clear, which
 * is what lets the compass rose sit behind the footage instead of over it.
 *
 * @param[in]     source  region of the frame to scale into the panel
 * @param[in]     mask    pixels of `source` that carry footage
 * @param[in,out] panel   panel drawn into
 * @exceptsafe basic
 */
void paste_panel(
    const cv::Mat& source,
    const cv::Mat& mask,
    cv::Mat& panel
) {
  cv::Mat scaled, scaled_mask;
  cv::resize(source, scaled, panel.size(), 0.0, 0.0, cv::INTER_AREA);
  cv::resize(mask, scaled_mask, panel.size(), 0.0, 0.0, cv::INTER_AREA);
  scaled.copyTo(panel, scaled_mask);
}

/**
 * Lay each view's orbit band into the panel, seen from straight above.
 *
 * One warp per view, since a single transform cannot face three differently
 * posed boards at once. The band is warped at twice the panel's resolution
 * and averaged down, which keeps the markers from aliasing the way sampling
 * the frame straight into place would.
 *
 * @param[in]     cleaned   frame with everything but the boards masked away
 * @param[in]     analysis  geometry, tracks and fits for this frame
 * @param[in]     views     rectified transform of every view
 * @param[in,out] panel     panel drawn into
 * @exceptsafe basic
 */
void paste_rectified_bands(
    const cv::Mat& cleaned,
    const Analysis& analysis,
    const PanelViews& views,
    cv::Mat& panel
) {
  const cv::Size dense(panel.cols * 2, panel.rows * 2);
  const cv::Matx33d supersample(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);
  cv::Mat warped, warped_mask, scaled, scaled_mask;
  for (int view = 0; view < VIEW_COUNT; ++view) {
    if (!views[view].fitted) continue;
    const cv::Mat& image_to_board = analysis.geometry[view].image_to_board;
    const cv::Mat mask = board_annulus_mask(
        image_to_board,
        analysis.circles[view].center,
        cleaned.size()
    );
    const cv::Mat to_dense(supersample * views[view].to_panel);
    cv::warpPerspective(cleaned, warped, to_dense, dense);
    cv::warpPerspective(mask, warped_mask, to_dense, dense, cv::INTER_NEAREST);
    cv::resize(warped, scaled, panel.size(), 0.0, 0.0, cv::INTER_AREA);
    cv::resize(
        warped_mask,
        scaled_mask,
        panel.size(),
        0.0,
        0.0,
        cv::INTER_AREA
    );
    scaled.copyTo(panel, scaled_mask);
  }
}

/**
 * Read a clip's rate, length and frame size before the run proper.
 *
 * The frame size is taken from a decoded frame rather than from the
 * container's properties, which lie often enough to matter. A missing or
 * nonsensical rate falls back to 24, since the rate only converts frame
 * indices to seconds and a wrong one scales every latency alike.
 *
 * @param[in] input_path  clip to probe
 * @returns what was found
 * @throws std::runtime_error if it cannot be opened or decodes no frames
 * @exceptsafe basic
 */
VideoInfo probe_video(const std::string& input_path) {
  cv::VideoCapture capture = open_video(input_path);
  VideoInfo info;
  info.fps = capture.get(cv::CAP_PROP_FPS);
  if (info.fps <= 0.0) info.fps = 24.0;
  info.frame_count = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  cv::Mat frame;
  if (!capture.read(frame) || frame.empty())
    throw std::runtime_error("no frames decoded from " + input_path);
  info.size = frame.size();
  capture.release();
  return info;
}

/**
 * Decide how many boards the recording shows.
 *
 * Votes over the first frames, which sets the view names and count used for
 * the whole run.
 *
 * @param[in] input_path  recording to inspect
 * @throws std::runtime_error if the video cannot be read
 * @exceptsafe basic
 */
void detect_view_layout(const std::string& input_path) {
  Detector detector = make_detector();
  cv::VideoCapture capture = open_video(input_path);
  std::map<int, int> votes;
  cv::Mat frame;
  for (int i = 0; i < 15 && capture.read(frame) && !frame.empty(); ++i)
    ++votes[count_boards_in_frame(frame, detector)];
  capture.release();

  int boards = 2, best = 0;
  for (const auto& vote : votes) {
    if (vote.first < 2 || vote.second <= best) continue;
    boards = vote.first;
    best = vote.second;
  }
  configure_views(boards >= 3);
}

/**
 * Run the whole measurement pass over one recording.
 *
 * Decodes each frame, locates the boards, measures the diode, renders the
 * four panels and hands them to the encoder. The first and last half second
 * are skipped as warm-up.
 *
 * @param[in] input_path   recording to analyse
 * @param[in] output_path  panel video to write
 * @param[in] info         frame rate, size and frame count of the input
 * @throws std::runtime_error if the video cannot be read or written, or if no
 *         frame had every board fitted
 * @exceptsafe basic
 */
void main_stream(
    const std::string& input_path,
    const std::string& output_path,
    const VideoInfo& info
) {
  const std::filesystem::path parent =
      std::filesystem::path(output_path).parent_path();
  if (!parent.empty()) {
    std::error_code error;
    std::filesystem::create_directories(parent, error);
  }
  const cv::Size output_size(OUTPUT_WIDTH, OUTPUT_HEIGHT);
  BackgroundWriter writer;
  writer_open(writer, output_path, info.fps, output_size);

  Detector detector = make_detector();
  cv::VideoCapture capture = open_video(input_path);
  Panels panels = make_panels();
  Analysis analysis = make_analysis(info.fps);
  CropSmoother smoother;
  std::vector<CropBox> tracked;
  std::vector<PoseSmoother> poses(VIEW_COUNT);
  cv::Mat frame, cleaned, gray;

  const double clip_seconds = info.frame_count / info.fps;
  for (int decoded = 0, index = 0; capture.read(frame) && !frame.empty();
       ++decoded) {
    const double at = decoded / info.fps;
    if (at < WARMUP_SECONDS) continue;
    if (info.frame_count > 0 && at > clip_seconds - WARMUP_SECONDS) break;

    const cv::Size size = frame.size();
    if (cleaned.empty()) cleaned.create(size, CV_8UC3);
    analysis.index = index++;
    analysis.geometry = detect_view_geometry(frame, detector, tracked);
    for (int view = 0; view < VIEW_COUNT; ++view)
      if (fitted(analysis.geometry[view]))
        analysis.geometry[view].image_to_board =
            smooth_pose(poses[view], analysis.geometry[view].image_to_board);

    const cv::Mat boards = board_mask_for(analysis.geometry, size);
    cleaned.setTo(SITE_GREEN);
    frame.copyTo(cleaned, boards);
    cv::cvtColor(cleaned, gray, cv::COLOR_BGR2GRAY);
    track_diodes(cleaned, gray, analysis);

    const cv::Rect target = board_crop_rect(analysis.geometry, size);
    const cv::Rect crop = smooth_crop_rect(smoother, target, size, info.fps);
    const PanelViews cropped = cropped_views(analysis, panel_transform(crop));
    const PanelViews rectified = rectified_views(analysis);

    fill_panel(frame(crop), panels.raw);

    panels.overlay.setTo(SITE_GREEN);
    render_compasses(panels.overlay, analysis, cropped);
    paste_panel(cleaned(crop), boards(crop), panels.overlay);
    render_overlay(panels.overlay, analysis, cropped);

    panels.band.setTo(SITE_GREEN);
    render_compasses(panels.band, analysis, rectified);
    paste_rectified_bands(cleaned, analysis, rectified, panels.band);
    render_band(panels.band, analysis, rectified);

    render_numbers(panels.numbers_full, analysis);
    fill_panel(panels.numbers_full, panels.numbers);
    draw_panel_border(panels.stacked);
    writer_submit(writer, panels.stacked);
  }

  capture.release();
  writer_close(writer);
  if (analysis.paired == 0)
    throw std::runtime_error("no frames had every board fitted");
  std::printf("%.1f\n", analysis.result.latency_ms);
}

}

/**
 * Parse the arguments and run one clip.
 *
 * The body is a function-try-block, so anything thrown anywhere in the run
 * lands here: the tool has one failure mode from a caller's point of view, a
 * message on stderr and a non-zero exit, and no partial number is ever
 * printed on stdout.
 *
 * @param[in] argc  argument count
 * @param[in] argv  arguments; `--input` and `--output` are both required
 * @returns 0 when a latency was measured and written, 1 otherwise
 * @exceptsafe no-throw
 */
int main(
    int argc,
    char** argv
) try {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      INPUT_PATH = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      OUTPUT_PATH = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "usage: %s --input PATH --output PATH\n"
          "measures teleoperation stream latency from a recording that shows\n"
          "the headset passthrough and the robot camera views of one marker\n"
          "board carrying a spinning diode\n"
          "prints the latency in milliseconds on stdout, nothing else\n"
          "signed as camera - passthrough: positive means the camera view is\n"
          "behind the live passthrough by that much\n",
          argv[0]
      );
      return 0;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  cv::theRNG().state = RANDOM_SEED;
  mallopt(M_MMAP_THRESHOLD, 256 * 1024);
  mallopt(M_TRIM_THRESHOLD, 256 * 1024);
  const VideoInfo info = probe_video(INPUT_PATH);
  detect_view_layout(INPUT_PATH);
  main_stream(INPUT_PATH, OUTPUT_PATH, info);
  return 0;
} catch (const std::exception& failure) {
  std::fprintf(stderr, "%s\n", failure.what());
  return 1;
}
