$fn = 180;

disk_dia            = 75;
disk_h              = 1;
disk_hole_dia       = 5;
disk_hole_count     = 20;
disk_hole_edge_gap  = 5;
disk_hole_ring_r    = disk_dia / 2 - disk_hole_edge_gap - disk_hole_dia / 2;

battery_len    = 50;
battery_width  = 26.5;
fit_clearance  = 1.0;
wall_th        = 1;
wall_h         = 15;
wall_len       = battery_len + fit_clearance;
wall_y         = (battery_width + fit_clearance) / 2 + wall_th / 2;

stick_tip_x    = 250;
stick_w_base   = 15;
stick_w_tip    = 3;
stick_overlap  = 5;
stick_base_x   = disk_dia / 2 - stick_overlap;
tip_pad_dia    = 15;
tip_hole_dia   = 10.5;
rib_w          = 1;
rib_h_base     = 8;
rib_h_tip      = 0.5;
rib_base_x     = disk_dia / 2;
rib_tip_x      = stick_tip_x - tip_pad_dia / 2;

cut_x     = (-disk_dia / 2 + stick_tip_x + tip_pad_dia / 2) / 2;
cut_w     = 0.1;
cut_yaw   = 45;
cut_tilt  = 45;

module disk() {
    difference() {
        cylinder(h = disk_h, d = disk_dia);
        for (i = [0 : disk_hole_count - 1]) {
            angle = (i + 0.5) * 360 / disk_hole_count;
            translate([disk_hole_ring_r * cos(angle), disk_hole_ring_r * sin(angle), -1])
                cylinder(h = disk_h + 2, d = disk_hole_dia);
        }
    }
}

module battery_walls() {
    for (side = [1, -1])
        translate([0, side * wall_y, disk_h + wall_h / 2])
            cube([wall_len, wall_th, wall_h], center = true);
}

module stick() {
    difference() {
        union() {
            linear_extrude(height = disk_h)
                polygon([[stick_base_x,  stick_w_base / 2],
                         [stick_tip_x,   stick_w_tip / 2],
                         [stick_tip_x,  -stick_w_tip / 2],
                         [stick_base_x, -stick_w_base / 2]]);
            translate([stick_tip_x, 0, 0])
                cylinder(h = disk_h, d = tip_pad_dia);
            translate([0, rib_w / 2, disk_h])
                rotate([90, 0, 0])
                    linear_extrude(height = rib_w)
                        polygon([[rib_base_x, 0],
                                 [rib_tip_x,  0],
                                 [rib_tip_x,  rib_h_tip],
                                 [rib_base_x, rib_h_base]]);
        }
        translate([stick_tip_x, 0, -1])
            cylinder(h = disk_h + 2, d = tip_hole_dia);
    }
}

module rib_wall_braces() {
    for (side = [1, -1])
        hull() {
            translate([rib_base_x, 0, disk_h + rib_h_base / 2])
                cube([rib_w, rib_w, rib_h_base], center = true);
            translate([wall_len / 2, side * wall_y, disk_h + wall_h / 2])
                cube([wall_th, wall_th, wall_h], center = true);
        }
}

module at_cut_plane() {
    translate([cut_x, 0, disk_h])
        rotate([0, cut_tilt, cut_yaw])
            children();
}

module assembly() {
    difference() {
        union() {
            disk();
            battery_walls();
            stick();
            rib_wall_braces();
        }
        at_cut_plane()
            cube([cut_w, disk_dia + 20, (disk_h + wall_h) * 4], center = true);
    }
}

module half(side) {
    intersection() {
        assembly();
        at_cut_plane()
            translate([side == "left" ? -1000 : 0, -500, -500])
                cube([1000, 1000, 1000]);
    }
}

part = "all";

if      (part == "left")  half("left");
else if (part == "right") half("right");
else                      assembly();
