use super::*;

pub(super) fn accessibility_backend() -> &'static str {
    if env::var("AT_SPI_BUS_ADDRESS").is_ok_and(|value| !value.trim().is_empty())
        || (command_exists("busctl") && atspi_bus_address().is_ok())
    {
        "atspi_dbus"
    } else {
        "unavailable"
    }
}

pub(super) fn accessibility_tree(snapshot_id: &str, limit: usize) -> Result<Value> {
    match native_accessibility_tree(snapshot_id, limit) {
        Ok(value) => Ok(value),
        Err(native_error) if command_exists("busctl") => {
            accessibility_tree_fallback(limit, &native_error.to_string())
        }
        Err(error) => Ok(json!({
            "ok": false,
            "backend": "unavailable",
            "reason": "backend_unavailable",
            "error": error.to_string(),
            "items": [],
            "count": 0,
        })),
    }
}

pub(super) fn native_atspi_address() -> Result<String> {
    if let Ok(address) = env::var("AT_SPI_BUS_ADDRESS") {
        if !address.trim().is_empty() {
            return Ok(address);
        }
    }
    atspi_bus_address()
}

pub(super) fn native_accessibility_tree(snapshot_id: &str, limit: usize) -> Result<Value> {
    let address = native_atspi_address()?;
    let connection = zbus::blocking::connection::Builder::address(address.as_str())?
        .method_timeout(Duration::from_secs(2))
        .build()?;
    let root = AccessibilityRef {
        bus: "org.a11y.atspi.Registry".to_string(),
        path: "/org/a11y/atspi/accessible/root".to_string(),
        depth: 0,
        parent_id: None,
    };
    let mut queue = VecDeque::from([root]);
    let mut visited = HashSet::new();
    let mut items = Vec::new();
    let mut skipped = 0usize;

    while let Some(reference) = queue.pop_front() {
        if items.len() >= limit {
            break;
        }
        let key = format!("{}\0{}", reference.bus, reference.path);
        if !visited.insert(key) {
            continue;
        }
        match accessibility_element(&connection, snapshot_id, &reference) {
            Ok((item, children)) => {
                let parent_id = item.get("id").and_then(Value::as_str).map(str::to_string);
                items.push(item);
                queue.extend(children.into_iter().map(|(bus, path)| AccessibilityRef {
                    bus,
                    path,
                    depth: reference.depth.saturating_add(1),
                    parent_id: parent_id.clone(),
                }));
            }
            Err(_) => skipped += 1,
        }
    }

    Ok(json!({
        "ok": true,
        "backend": "atspi_dbus",
        "snapshot_id": snapshot_id,
        "content": "accessible_metadata_only",
        "count": items.len(),
        "items": items,
        "truncated": !queue.is_empty(),
        "skipped_defunct_or_unreachable": skipped,
    }))
}

#[derive(Debug)]
pub(super) struct AccessibilityRef {
    bus: String,
    path: String,
    depth: usize,
    parent_id: Option<String>,
}

pub(super) fn accessibility_element(
    connection: &zbus::blocking::Connection,
    snapshot_id: &str,
    reference: &AccessibilityRef,
) -> Result<(Value, Vec<(String, String)>)> {
    let accessible = zbus::blocking::Proxy::new(
        connection,
        reference.bus.as_str(),
        reference.path.as_str(),
        "org.a11y.atspi.Accessible",
    )?;
    let name = accessible
        .get_property::<String>("Name")
        .unwrap_or_default();
    let description = accessible
        .get_property::<String>("Description")
        .unwrap_or_default();
    let role = accessible
        .call::<_, _, String>("GetRoleName", &())
        .unwrap_or_else(|_| "unknown".to_string());
    let state_ids = accessible
        .call::<_, _, Vec<u32>>("GetState", &())
        .unwrap_or_default();
    let interfaces = accessible
        .call::<_, _, Vec<String>>("GetInterfaces", &())
        .unwrap_or_default();
    let children = accessible
        .call::<_, _, Vec<(String, zbus::zvariant::OwnedObjectPath)>>("GetChildren", &())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(bus, path)| {
            let path = path.to_string();
            (path != "/org/a11y/atspi/null").then_some((bus, path))
        })
        .collect::<Vec<_>>();
    let bounds = accessibility_bounds(connection, reference, &interfaces);
    let actions = accessibility_actions(connection, reference, &interfaces);
    let target_id = accessibility_target_id(snapshot_id, &reference.bus, &reference.path);

    Ok((
        json!({
            "id": target_id,
            "snapshot_id": snapshot_id,
            "depth": reference.depth,
            "parent_id": reference.parent_id,
            "name": bounded_accessible_text(&name),
            "description": bounded_accessible_text(&description),
            "role": role,
            "state_ids": state_ids,
            "interfaces": interfaces,
            "actions": actions,
            "bounds": bounds,
            "child_count": children.len(),
            "backend_ref": {
                "bus": reference.bus,
                "path": reference.path,
            },
        }),
        children,
    ))
}

pub(super) fn accessibility_bounds(
    connection: &zbus::blocking::Connection,
    reference: &AccessibilityRef,
    interfaces: &[String],
) -> Option<Value> {
    if !interfaces.iter().any(|item| item.ends_with(".Component")) {
        return None;
    }
    let proxy = zbus::blocking::Proxy::new(
        connection,
        reference.bus.as_str(),
        reference.path.as_str(),
        "org.a11y.atspi.Component",
    )
    .ok()?;
    let (x, y, width, height): (i32, i32, i32, i32) = proxy.call("GetExtents", &0u32).ok()?;
    Some(json!({ "x": x, "y": y, "width": width, "height": height }))
}

pub(super) fn accessibility_actions(
    connection: &zbus::blocking::Connection,
    reference: &AccessibilityRef,
    interfaces: &[String],
) -> Vec<String> {
    if !interfaces.iter().any(|item| item.ends_with(".Action")) {
        return Vec::new();
    }
    let Ok(proxy) = zbus::blocking::Proxy::new(
        connection,
        reference.bus.as_str(),
        reference.path.as_str(),
        "org.a11y.atspi.Action",
    ) else {
        return Vec::new();
    };
    let count = proxy
        .call::<_, _, i32>("GetNActions", &())
        .unwrap_or(0)
        .clamp(0, 32);
    (0..count)
        .filter_map(|index| proxy.call::<_, _, String>("GetName", &index).ok())
        .map(|name| bounded_accessible_text(&name))
        .filter(|name| !name.is_empty())
        .collect()
}

pub(super) fn accessibility_target_id(snapshot_id: &str, bus: &str, path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(snapshot_id.as_bytes());
    hasher.update([0]);
    hasher.update(bus.as_bytes());
    hasher.update([0]);
    hasher.update(path.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format!("ui-{}", &digest[..16])
}

pub(super) fn bounded_accessible_text(value: &str) -> String {
    value.chars().take(256).collect()
}

pub(super) fn dialog_inventory(snapshot_id: &str, limit: usize) -> Result<Value> {
    let tree_limit = limit.saturating_mul(10).clamp(1, 200);
    let tree = accessibility_tree(snapshot_id, tree_limit)?;
    if tree.get("ok").and_then(Value::as_bool) != Some(true) {
        return Ok(json!({
            "ok": false,
            "backend": tree.get("backend").cloned().unwrap_or(json!("unavailable")),
            "reason": tree.get("reason").cloned().unwrap_or(json!("backend_unavailable")),
            "error": tree.get("error").cloned(),
            "items": [],
            "count": 0,
        }));
    }
    let mut dialogs = dialog_items(
        tree.get("items")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default(),
        limit,
    );
    let total = dialogs.len();
    dialogs.truncate(limit);
    Ok(json!({
        "ok": true,
        "backend": tree.get("backend").cloned().unwrap_or(json!("atspi_dbus")),
        "snapshot_id": snapshot_id,
        "count": total.min(limit),
        "items": dialogs,
        "truncated": total > limit,
        "tree_truncated": tree.get("truncated").cloned().unwrap_or(json!(false)),
    }))
}

pub(super) fn dialog_items(items: &[Value], control_limit: usize) -> Vec<Value> {
    let parents = items
        .iter()
        .filter_map(|item| {
            Some((
                item.get("id")?.as_str()?,
                item.get("parent_id")
                    .and_then(Value::as_str)
                    .filter(|parent_id| !parent_id.is_empty()),
            ))
        })
        .collect::<HashMap<_, _>>();

    items
        .iter()
        .filter_map(|item| {
            let role = item.get("role").and_then(Value::as_str)?;
            let kind = dialog_kind(role)?;
            let dialog_id = item.get("id").and_then(Value::as_str)?;
            let controls = items
                .iter()
                .filter(|candidate| {
                    candidate.get("id").and_then(Value::as_str) != Some(dialog_id)
                        && accessibility_descends_from(candidate, dialog_id, &parents)
                })
                .take(control_limit)
                .cloned()
                .collect::<Vec<_>>();
            let mut dialog = item.clone();
            let object = dialog.as_object_mut()?;
            object.insert("dialog_kind".to_string(), json!(kind));
            object.insert("control_count".to_string(), json!(controls.len()));
            object.insert("controls".to_string(), json!(controls));
            Some(dialog)
        })
        .collect()
}

pub(super) fn accessibility_descends_from(
    item: &Value,
    ancestor_id: &str,
    parents: &HashMap<&str, Option<&str>>,
) -> bool {
    let mut current = item.get("parent_id").and_then(Value::as_str);
    let mut visited = HashSet::new();
    while let Some(id) = current {
        if id == ancestor_id {
            return true;
        }
        if !visited.insert(id) {
            return false;
        }
        current = parents.get(id).copied().flatten();
    }
    false
}

pub(super) fn dialog_kind(role: &str) -> Option<&'static str> {
    let normalized = role.trim().to_ascii_lowercase().replace(['_', '-'], " ");
    if normalized.contains("file chooser") || normalized.contains("file picker") {
        Some("file_picker")
    } else if normalized == "alert" || normalized.contains("alert dialog") {
        Some("alert")
    } else if normalized == "dialog" || normalized.ends_with(" dialog") {
        Some("dialog")
    } else {
        None
    }
}

pub(super) fn ui_element_action(
    action: &str,
    target_id: &str,
    value: Option<&str>,
    backend_bus: Option<&str>,
    backend_path: Option<&str>,
) -> Result<Value> {
    let bus =
        backend_bus.ok_or_else(|| anyhow::anyhow!("{action} requires an observed UI element"))?;
    let path =
        backend_path.ok_or_else(|| anyhow::anyhow!("{action} requires an observed UI element"))?;
    validate_accessibility_reference(bus, path)?;
    let address = native_atspi_address()?;
    let connection = zbus::blocking::connection::Builder::address(address.as_str())?
        .method_timeout(Duration::from_secs(2))
        .build()?;

    match action {
        "focus_element" => focus_accessibility_element(&connection, target_id, bus, path),
        "invoke_element" => invoke_accessibility_element(
            &connection,
            target_id,
            bus,
            path,
            value.ok_or_else(|| {
                anyhow::anyhow!("invoke_element requires an advertised action name")
            })?,
        ),
        "set_field_text" => set_accessibility_text(
            &connection,
            target_id,
            bus,
            path,
            value.ok_or_else(|| anyhow::anyhow!("set_field_text requires text"))?,
        ),
        _ => unreachable!("semantic action was validated by caller"),
    }
}

pub(super) fn validate_accessibility_reference(bus: &str, path: &str) -> Result<()> {
    zbus::names::BusName::try_from(bus)
        .map_err(|_| anyhow::anyhow!("Invalid accessibility bus name"))?;
    zbus::zvariant::ObjectPath::try_from(path)
        .map_err(|_| anyhow::anyhow!("Invalid accessibility object path"))?;
    Ok(())
}

pub(super) fn focus_accessibility_element(
    connection: &zbus::blocking::Connection,
    target_id: &str,
    bus: &str,
    path: &str,
) -> Result<Value> {
    const ATSPI_STATE_FOCUSED: u32 = 12;
    let before = accessibility_state(connection, bus, path);
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Component")?;
    let dispatched: bool = proxy.call("GrabFocus", &())?;
    thread::sleep(Duration::from_millis(75));
    let after = accessibility_state(connection, bus, path);
    let verified = dispatched
        && after
            .as_ref()
            .is_some_and(|states| states.contains(&ATSPI_STATE_FOCUSED));
    Ok(json!({
        "ok": dispatched,
        "tool": "desktop_action",
        "action": "focus_element",
        "target": target_id,
        "backend": "atspi_dbus",
        "before": { "state_ids": before },
        "after": { "state_ids": after },
        "verified": verified,
        "verification": if verified { "confirmed" } else { "not_confirmed" },
        "reason": if dispatched && !verified { Some("verification_failed") } else if !dispatched { Some("dispatch_rejected") } else { None },
    }))
}

pub(super) fn invoke_accessibility_element(
    connection: &zbus::blocking::Connection,
    target_id: &str,
    bus: &str,
    path: &str,
    requested_action: &str,
) -> Result<Value> {
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Action")?;
    let count = proxy.call::<_, _, i32>("GetNActions", &())?.clamp(0, 32);
    let mut index = None;
    let mut advertised = Vec::new();
    for candidate in 0..count {
        if let Ok(name) = proxy.call::<_, _, String>("GetName", &candidate) {
            if name == requested_action {
                index = Some(candidate);
            }
            advertised.push(name);
        }
    }
    let index = index.ok_or_else(|| {
        anyhow::anyhow!(
            "Element does not advertise action {requested_action:?}; available: {}",
            advertised.join(", ")
        )
    })?;
    let before = accessibility_state(connection, bus, path);
    let dispatched: bool = proxy.call("DoAction", &index)?;
    thread::sleep(Duration::from_millis(100));
    let after = accessibility_state(connection, bus, path);
    let observable_change = before != after || (before.is_some() && after.is_none());
    let verified = dispatched && observable_change;
    Ok(json!({
        "ok": dispatched,
        "tool": "desktop_action",
        "action": "invoke_element",
        "target": target_id,
        "element_action": requested_action,
        "backend": "atspi_dbus",
        "before": { "state_ids": before },
        "after": { "state_ids": after },
        "verified": verified,
        "verification": if verified { "confirmed" } else if dispatched { "dispatch_confirmed" } else { "failed" },
        "reason": if dispatched && !verified { Some("effect_not_observable_on_target") } else if !dispatched { Some("dispatch_rejected") } else { None },
    }))
}

pub(super) fn set_accessibility_text(
    connection: &zbus::blocking::Connection,
    target_id: &str,
    bus: &str,
    path: &str,
    text: &str,
) -> Result<Value> {
    if text.len() > 64 * 1024 {
        return Err(anyhow::anyhow!(
            "set_field_text is limited to 65536 UTF-8 bytes"
        ));
    }
    let before_hash = accessibility_text_hash(connection, bus, path);
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.EditableText")?;
    let dispatched: bool = proxy.call("SetTextContents", &text)?;
    thread::sleep(Duration::from_millis(75));
    let after_hash = accessibility_text_hash(connection, bus, path);
    let expected_hash = sha256_text(text);
    let verified = dispatched && after_hash.as_deref() == Some(expected_hash.as_str());
    Ok(json!({
        "ok": dispatched,
        "tool": "desktop_action",
        "action": "set_field_text",
        "target": target_id,
        "backend": "atspi_dbus",
        "value": { "redacted": true, "sha256": expected_hash, "bytes": text.len(), "characters": text.chars().count() },
        "before": { "sha256": before_hash },
        "after": { "sha256": after_hash },
        "verified": verified,
        "verification": if verified { "confirmed" } else { "not_confirmed" },
        "reason": if dispatched && !verified { Some("verification_failed") } else if !dispatched { Some("dispatch_rejected") } else { None },
    }))
}

pub(super) fn accessibility_state(
    connection: &zbus::blocking::Connection,
    bus: &str,
    path: &str,
) -> Option<Vec<u32>> {
    let proxy =
        zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Accessible").ok()?;
    proxy.call("GetState", &()).ok()
}

pub(super) fn accessibility_text_hash(
    connection: &zbus::blocking::Connection,
    bus: &str,
    path: &str,
) -> Option<String> {
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Text").ok()?;
    let value: String = proxy.call("GetText", &(0i32, -1i32)).ok()?;
    Some(sha256_text(&value))
}

pub(super) fn sha256_text(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(super) fn accessibility_tree_fallback(limit: usize, native_error: &str) -> Result<Value> {
    let address = match atspi_bus_address() {
        Ok(address) => address,
        Err(error) => {
            return Ok(json!({
                "ok": false,
                "backend": "unavailable",
                "reason": "backend_unavailable",
                "error": error.to_string(),
                "items": [],
                "count": 0,
            }));
        }
    };
    let address_arg = format!("--address={address}");
    let output = run_capture_dynamic(&[
        "busctl",
        &address_arg,
        "--no-pager",
        "--list",
        "tree",
        "org.a11y.atspi.Registry",
        "/org/a11y/atspi/accessible/root",
    ])?;
    let items: Vec<Value> = output
        .stdout
        .lines()
        .take(limit)
        .filter_map(accessibility_tree_item)
        .collect();
    Ok(json!({
        "ok": output.status == 0,
        "backend": "atspi_busctl_fallback",
        "content": "object_paths_only",
        "count": items.len(),
        "items": items,
        "truncated": output.stdout.lines().count() > limit,
        "stderr": output.stderr,
        "native_backend_error": native_error,
    }))
}

pub(super) fn atspi_bus_address() -> Result<String> {
    if !command_exists("busctl") {
        return Err(anyhow::anyhow!("AT-SPI observation requires busctl."));
    }
    let output = run_capture_dynamic(&[
        "busctl",
        "--user",
        "call",
        "org.a11y.Bus",
        "/org/a11y/bus",
        "org.a11y.Bus",
        "GetAddress",
    ])?;
    if output.status != 0 {
        return Err(anyhow::anyhow!(
            "AT-SPI bus is unavailable: {}",
            if output.stderr.is_empty() {
                output.stdout
            } else {
                output.stderr
            }
        ));
    }
    parse_busctl_string(&output.stdout)
        .ok_or_else(|| anyhow::anyhow!("AT-SPI bus address could not be parsed."))
}

pub(super) fn accessibility_tree_item(line: &str) -> Option<Value> {
    let path = line.trim();
    if !path.starts_with("/org/a11y/atspi/accessible") {
        return None;
    }
    Some(json!({
        "path": path,
        "kind": "accessible_object",
    }))
}

pub(super) fn parse_busctl_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let start = trimmed.find('"')?;
    let end = trimmed.rfind('"')?;
    if end <= start {
        return None;
    }
    let text = &trimmed[start + 1..end];
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}
