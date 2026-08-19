use super::*;

pub(super) fn display_backend() -> &'static str {
    if command_exists("xrandr") {
        "xrandr"
    } else {
        "unavailable"
    }
}

pub(super) fn display_inventory(limit: usize) -> Result<Value> {
    if display_backend() == "unavailable" {
        return Ok(json!({
            "ok": false,
            "backend": "unavailable",
            "reason": "backend_unavailable",
            "error": "Display observation requires xrandr.",
            "items": [],
            "count": 0,
        }));
    }

    let output = run_capture_dynamic(&["xrandr", "--query"])?;
    let mut items = parse_xrandr_displays(&output.stdout);
    let total = items.len();
    items.truncate(limit);
    Ok(json!({
        "ok": output.status == 0,
        "backend": "xrandr",
        "count": items.len(),
        "items": items,
        "truncated": total > limit,
        "stderr": output.stderr,
    }))
}

pub(super) fn parse_xrandr_displays(output: &str) -> Vec<Value> {
    output.lines().filter_map(parse_xrandr_display).collect()
}

pub(super) fn parse_xrandr_display(line: &str) -> Option<Value> {
    let mut fields = line.split_whitespace();
    let name = fields.next()?;
    let status = fields.next()?;
    if !matches!(status, "connected" | "disconnected") {
        return None;
    }
    let mut primary = false;
    let mut geometry = None;
    for field in fields {
        primary |= field == "primary";
        geometry = geometry.or_else(|| parse_xrandr_geometry(field));
    }
    Some(json!({
        "name": name,
        "status": status,
        "connected": status == "connected",
        "primary": primary,
        "geometry": geometry.map(|(width, height, x, y)| json!({
            "width": width,
            "height": height,
            "x": x,
            "y": y,
        })),
    }))
}

pub(super) fn parse_xrandr_geometry(value: &str) -> Option<(u64, u64, i64, i64)> {
    let (width, rest) = value.split_once('x')?;
    let offset_start = rest.find(['+', '-'])?;
    let (height, offsets) = rest.split_at(offset_start);
    let second_offset = offsets[1..].find(['+', '-'])? + 1;
    let (x, y) = offsets.split_at(second_offset);
    Some((
        width.parse().ok()?,
        height.parse().ok()?,
        x.parse().ok()?,
        y.parse().ok()?,
    ))
}
