use super::*;

pub(super) fn keyboard_backend() -> &'static str {
    if command_exists("wtype") {
        "wtype"
    } else if command_exists("xdotool") {
        "xdotool"
    } else {
        "unavailable"
    }
}

pub(super) fn pointer_backend() -> &'static str {
    if command_exists("xdotool") {
        "xdotool"
    } else {
        "unavailable"
    }
}

pub(super) fn pointer_action(
    action: &str,
    target: Option<&str>,
    value: Option<&str>,
) -> Result<Value> {
    match action {
        "mouse_click" => {
            let (x, y) = parse_pointer_coordinates(required_target(action, target)?)?;
            let button = pointer_button(value)?;
            let x_arg = x.to_string();
            let y_arg = y.to_string();
            let button_arg = button.to_string();
            let argv = [
                "xdotool",
                "mousemove",
                "--sync",
                x_arg.as_str(),
                y_arg.as_str(),
                "click",
                button_arg.as_str(),
            ];
            pointer_receipt(action, json!({ "x": x, "y": y, "button": button }), &argv)
        }
        "scroll" => {
            let steps = parse_scroll_steps(value)?;
            let button = if steps < 0 { "5" } else { "4" };
            let repetitions = steps.unsigned_abs().min(25);
            let mut argv = Vec::with_capacity(1 + repetitions as usize * 2);
            argv.push(String::from("xdotool"));
            for _ in 0..repetitions {
                argv.push(String::from("click"));
                argv.push(String::from(button));
            }
            pointer_receipt(
                action,
                json!({ "steps": steps, "button": button, "repetitions": repetitions }),
                &argv,
            )
        }
        _ => Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "unsupported_action",
            "error": format!("Unsupported pointer action: {action}"),
        })),
    }
}

pub(super) fn pointer_receipt<T: AsRef<std::ffi::OsStr>>(
    action: &str,
    value: Value,
    argv: &[T],
) -> Result<Value> {
    if !command_exists("xdotool") {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "backend": "xdotool",
            "reason": "dependency_unavailable",
            "error": "Pointer input requires xdotool.",
        }));
    }
    let before = active_window()?;
    let output = run_capture_dynamic(argv)?;
    thread::sleep(Duration::from_millis(50));
    let after = active_window()?;
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "backend": "xdotool",
        "value": value,
        "before": before,
        "after": after,
        "verified": false,
        "verification": if output.status == 0 { "dispatched_unverified" } else { "failed" },
        "exit_code": output.status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn parse_pointer_coordinates(value: &str) -> Result<(i32, i32)> {
    let Some((raw_x, raw_y)) = value.split_once(',') else {
        return Err(anyhow::anyhow!(
            "mouse_click target must be coordinates formatted as x,y"
        ));
    };
    let x = raw_x
        .trim()
        .parse::<i32>()
        .map_err(|_| anyhow::anyhow!("mouse_click x coordinate must be an integer"))?;
    let y = raw_y
        .trim()
        .parse::<i32>()
        .map_err(|_| anyhow::anyhow!("mouse_click y coordinate must be an integer"))?;
    if !(0..=100_000).contains(&x) || !(0..=100_000).contains(&y) {
        return Err(anyhow::anyhow!(
            "mouse_click coordinates must be between 0 and 100000"
        ));
    }
    Ok((x, y))
}

pub(super) fn pointer_button(value: Option<&str>) -> Result<u8> {
    let raw = value.unwrap_or("1").trim();
    let button = raw
        .parse::<u8>()
        .map_err(|_| anyhow::anyhow!("mouse_click value must be a button number"))?;
    if !(1..=3).contains(&button) {
        return Err(anyhow::anyhow!("mouse_click button must be 1, 2, or 3"));
    }
    Ok(button)
}

pub(super) fn parse_scroll_steps(value: Option<&str>) -> Result<i32> {
    let raw = value.unwrap_or("-3").trim();
    let steps = raw
        .parse::<i32>()
        .map_err(|_| anyhow::anyhow!("scroll value must be a signed step count"))?;
    if steps == 0 || !(-25..=25).contains(&steps) {
        return Err(anyhow::anyhow!(
            "scroll value must be between -25 and 25, excluding 0"
        ));
    }
    Ok(steps)
}

pub(super) fn keyboard_action(action: &str, value: Option<&str>) -> Result<Value> {
    let raw = value.ok_or_else(|| anyhow::anyhow!("{action} requires --value"))?;
    match action {
        "send_key" => {
            let key = normalize_key_spec(raw)?;
            keyboard_receipt(
                action,
                &key,
                "xdotool",
                &["xdotool", "key", "--clearmodifiers", &key],
            )
        }
        "type_text" => {
            if raw.len() > 4000 {
                return Err(anyhow::anyhow!(
                    "type_text value must be 4000 bytes or fewer"
                ));
            }
            match keyboard_backend() {
                "wtype" => keyboard_receipt(action, raw, "wtype", &["wtype", raw]),
                "xdotool" => keyboard_receipt(
                    action,
                    raw,
                    "xdotool",
                    &[
                        "xdotool",
                        "type",
                        "--clearmodifiers",
                        "--delay",
                        "0",
                        "--",
                        raw,
                    ],
                ),
                _ => Ok(json!({
                    "ok": false,
                    "tool": "desktop_action",
                    "action": action,
                    "reason": "dependency_unavailable",
                    "error": "Keyboard text input requires wtype or xdotool.",
                })),
            }
        }
        _ => Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "unsupported_action",
            "error": format!("Unsupported keyboard action: {action}"),
        })),
    }
}

pub(super) fn keyboard_receipt(
    action: &str,
    value: &str,
    backend: &str,
    argv: &[&str],
) -> Result<Value> {
    let Some((program, _)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty keyboard command"));
    };
    if !command_exists(program) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "backend": backend,
            "reason": "dependency_unavailable",
            "error": format!("Required keyboard command `{program}` is not installed."),
        }));
    }
    let before = active_window()?;
    let output = run_capture_dynamic(argv)?;
    thread::sleep(Duration::from_millis(50));
    let after = active_window()?;
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "backend": backend,
        "value": keyboard_value_receipt(action, value),
        "before": before,
        "after": after,
        "verified": false,
        "verification": if output.status == 0 { "dispatched_unverified" } else { "failed" },
        "exit_code": output.status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn keyboard_value_receipt(action: &str, value: &str) -> Value {
    if action == "type_text" {
        json!({
            "byte_count": value.len(),
            "char_count": value.chars().count(),
            "content_returned": false,
        })
    } else {
        json!({
            "key": value,
            "content_returned": true,
        })
    }
}

pub(super) fn normalize_key_spec(value: &str) -> Result<String> {
    let normalized = value.trim();
    if normalized.is_empty() || normalized.len() > 80 {
        return Err(anyhow::anyhow!(
            "send_key value must be a non-empty key spec"
        ));
    }
    if !normalized
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '+' | ':' | '.'))
    {
        return Err(anyhow::anyhow!(
            "send_key accepts only key names and modifiers, such as ctrl+l or Alt+Tab"
        ));
    }
    Ok(normalized.to_string())
}
