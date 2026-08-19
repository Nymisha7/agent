use super::{
    accessibility_target_id, accessibility_tree_item, bounded_accessible_text, desktop_action,
    desktop_capabilities, desktop_observe, dialog_items, dialog_kind, hex_decode, hex_encode,
    parse_busctl_string, parse_pactl_mute, parse_pactl_volume_percent, parse_xrandr_displays,
    windows_host_volume_script,
};
use serde_json::{json, Value};

#[test]
fn hex_codec_roundtrips_every_byte_value() {
    let bytes = (u8::MIN..=u8::MAX).collect::<Vec<_>>();

    assert_eq!(hex_decode(&hex_encode(&bytes)).unwrap(), bytes);
}

#[test]
fn desktop_capabilities_reports_runtime_actions() {
    let capabilities = desktop_capabilities().expect("desktop capabilities");

    assert_eq!(capabilities.get("ok").and_then(Value::as_bool), Some(true));
    assert_eq!(
        capabilities.get("tool").and_then(Value::as_str),
        Some("desktop_capabilities")
    );
    assert!(capabilities
        .get("actions")
        .and_then(Value::as_array)
        .is_some_and(|actions| actions
            .iter()
            .any(|action| action.get("action").and_then(Value::as_str) == Some("set_volume"))));
    assert!(capabilities
        .get("backends")
        .and_then(Value::as_object)
        .is_some());
    let actions = capabilities
        .get("actions")
        .and_then(Value::as_array)
        .expect("desktop actions");
    for action_name in [
        "launch_application",
        "open_path",
        "open_path_in_application",
        "focus_window",
        "close_window",
    ] {
        let action = actions
            .iter()
            .find(|action| action.get("action").and_then(Value::as_str) == Some(action_name))
            .expect("direct desktop action");
        assert_eq!(action.get("safety").and_then(Value::as_str), Some("direct"));
    }
}

#[test]
fn opens_an_existing_path_with_a_selected_application() {
    let result = desktop_action(
        "open_path_in_application",
        std::env::temp_dir().to_str(),
        Some("true"),
        None,
        None,
    )
    .expect("desktop action");

    assert_eq!(result.get("ok").and_then(Value::as_bool), Some(true));
    assert_eq!(
        result.get("application").and_then(Value::as_str),
        Some("true")
    );
    assert_eq!(
        result.get("verification_scope").and_then(Value::as_str),
        Some("application_invocation")
    );
}

#[test]
fn open_path_in_application_rejects_shell_shaped_executables() {
    let error = desktop_action(
        "open_path_in_application",
        std::env::temp_dir().to_str(),
        Some("code;touch"),
        None,
        None,
    )
    .expect_err("unsafe executable must be rejected");

    assert!(error.to_string().contains("executable identifier"));
}

#[test]
fn desktop_observe_reports_versioned_read_only_snapshot() {
    let snapshot = desktop_observe("active_window", 5).expect("desktop observation");

    assert_eq!(
        snapshot.get("tool").and_then(Value::as_str),
        Some("desktop_observe")
    );
    assert_eq!(
        snapshot.get("schema_version").and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        snapshot.get("scope").and_then(Value::as_str),
        Some("active_window")
    );
    assert!(snapshot
        .get("snapshot_id")
        .and_then(Value::as_str)
        .is_some());
    assert!(snapshot
        .get("backends")
        .and_then(Value::as_object)
        .is_some());
    assert!(snapshot
        .get("limitations")
        .and_then(Value::as_array)
        .is_some());
}

#[test]
fn parses_atspi_busctl_string_reply() {
    assert_eq!(
        parse_busctl_string(r#"s "unix:path=/run/user/1000/at-spi/bus_0""#),
        Some("unix:path=/run/user/1000/at-spi/bus_0".to_string())
    );
    assert_eq!(parse_busctl_string("s \"\""), None);
    assert_eq!(parse_busctl_string("not a quoted reply"), None);
}

#[test]
fn accessibility_tree_item_accepts_only_atspi_paths() {
    let item =
        accessibility_tree_item("/org/a11y/atspi/accessible/root").expect("expected atspi item");

    assert_eq!(
        item.get("path").and_then(Value::as_str),
        Some("/org/a11y/atspi/accessible/root")
    );
    assert_eq!(
        item.get("kind").and_then(Value::as_str),
        Some("accessible_object")
    );
    assert!(accessibility_tree_item("├─/unrelated/path").is_none());
}

#[test]
fn accessibility_target_ids_are_snapshot_bound_and_deterministic() {
    let first = accessibility_target_id("desktop-1", ":1.42", "/org/example/button/1");
    let repeated = accessibility_target_id("desktop-1", ":1.42", "/org/example/button/1");
    let newer = accessibility_target_id("desktop-2", ":1.42", "/org/example/button/1");

    assert_eq!(first, repeated);
    assert_ne!(first, newer);
    assert!(first.starts_with("ui-"));
}

#[test]
fn accessibility_text_is_bounded_by_characters() {
    let text = format!("{}é", "a".repeat(300));
    let bounded = bounded_accessible_text(&text);

    assert_eq!(bounded.chars().count(), 256);
    assert!(bounded.is_char_boundary(bounded.len()));
}

#[test]
fn classifies_protocol_dialog_roles_without_application_names() {
    assert_eq!(dialog_kind("dialog"), Some("dialog"));
    assert_eq!(dialog_kind("alert dialog"), Some("alert"));
    assert_eq!(dialog_kind("file chooser"), Some("file_picker"));
    assert_eq!(dialog_kind("push button"), None);
}

#[test]
fn groups_dialog_controls_by_accessibility_parentage() {
    let items = vec![
        json!({"id": "root", "parent_id": null, "role": "application"}),
        json!({"id": "dialog", "parent_id": "root", "role": "file chooser"}),
        json!({"id": "panel", "parent_id": "dialog", "role": "panel"}),
        json!({"id": "name", "parent_id": "panel", "role": "text", "name": "Name"}),
        json!({"id": "save", "parent_id": "dialog", "role": "push button", "name": "Save"}),
        json!({"id": "outside", "parent_id": "root", "role": "push button"}),
    ];

    let dialogs = dialog_items(&items, 10);

    assert_eq!(dialogs.len(), 1);
    assert_eq!(
        dialogs[0].get("dialog_kind").and_then(Value::as_str),
        Some("file_picker")
    );
    let controls = dialogs[0]
        .get("controls")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(controls.len(), 3);
    assert!(controls
        .iter()
        .any(|item| item.get("id").and_then(Value::as_str) == Some("name")));
    assert!(!controls
        .iter()
        .any(|item| item.get("id").and_then(Value::as_str) == Some("outside")));
}

#[test]
fn parses_xrandr_display_state_and_geometry() {
    let displays = parse_xrandr_displays(
            "eDP-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis)\nHDMI-1 disconnected (normal left inverted right x axis y axis)\nDP-1 connected 2560x1440-2560+0 (normal left inverted right x axis y axis)",
        );

    assert_eq!(displays.len(), 3);
    assert_eq!(
        displays[0].get("name").and_then(Value::as_str),
        Some("eDP-1")
    );
    assert_eq!(
        displays[0].get("primary").and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        displays[0]
            .pointer("/geometry/width")
            .and_then(Value::as_u64),
        Some(1920)
    );
    assert_eq!(
        displays[1].get("connected").and_then(Value::as_bool),
        Some(false)
    );
    assert!(displays[1].get("geometry").is_some_and(Value::is_null));
    assert_eq!(
        displays[2].pointer("/geometry/x").and_then(Value::as_i64),
        Some(-2560)
    );
}

#[test]
fn parses_pactl_default_sink_state() {
    assert_eq!(
        parse_pactl_volume_percent(
            "Volume: front-left: 32768 / 50% / -18.06 dB, front-right: 32768 / 50% / -18.06 dB"
        ),
        Some(50)
    );
    assert_eq!(parse_pactl_mute("Mute: yes"), Some(true));
    assert_eq!(parse_pactl_mute("Mute: no"), Some(false));
    assert_eq!(parse_pactl_volume_percent("invalid"), None);
    assert_eq!(parse_pactl_mute("Mute: unknown"), None);
}

#[test]
fn launch_verification_requires_real_evidence() {
    let before = json!({
        "process": null,
        "windows": {"ok": false, "count": 0, "ids": [], "titles": []},
    });
    let after = json!({
        "process": {"running": true},
        "windows": {"ok": false, "count": 0, "ids": [], "titles": []},
    });

    assert!(!super::launch_observation_changed(&before, &after, false));
    assert!(super::launch_observation_changed(&before, &after, true));
}

#[test]
fn launch_verification_accepts_window_change() {
    let before = json!({
        "windows": {"ok": true, "count": 1, "ids": ["1"], "titles": ["Shell"]},
    });
    let after = json!({
        "windows": {"ok": true, "count": 2, "ids": ["1", "2"], "titles": ["Shell", "Browser"]},
    });

    assert!(super::launch_observation_changed(&before, &after, false));
}

#[test]
fn window_ids_are_normalized_without_accepting_shell_shapes() {
    assert_eq!(
        super::normalize_window_id("0x03a00007").as_deref(),
        Some("0x3a00007")
    );
    assert_eq!(
        super::normalize_window_id("60817415").as_deref(),
        Some("0x3a00007")
    );
    assert!(super::normalize_window_id("0x03a00007;rm").is_none());
    assert!(super::normalize_window_id("").is_none());
}

#[test]
fn window_state_verification_reads_net_wm_state_atoms() {
    let state = json!({
        "net_wm_state": {
            "states": [
                "_NET_WM_STATE_MAXIMIZED_VERT",
                "_NET_WM_STATE_MAXIMIZED_HORZ"
            ]
        }
    });

    assert!(super::window_has_state(
        &state,
        "_NET_WM_STATE_MAXIMIZED_VERT"
    ));
    assert!(!super::window_has_state(&state, "_NET_WM_STATE_HIDDEN"));
}

#[test]
fn window_action_requires_verified_postcondition_for_success() {
    assert_eq!(
        super::window_action_outcome(0, false),
        (false, "not_confirmed", Some("verification_failed"))
    );
    assert_eq!(
        super::window_action_outcome(0, true),
        (true, "confirmed", None)
    );
    assert_eq!(
        super::window_action_outcome(1, false),
        (false, "failed", Some("command_failed"))
    );
}

#[test]
fn clipboard_metadata_never_returns_content() {
    let metadata = super::clipboard_metadata().expect("clipboard metadata");

    assert_eq!(
        metadata.get("content_returned").and_then(Value::as_bool),
        Some(false)
    );
}

#[test]
fn clipboard_write_requires_value_before_backend_selection() {
    let error = super::clipboard_write_action("clipboard_write", None)
        .expect_err("missing clipboard value should fail");

    assert!(error.to_string().contains("requires --value"));
}

#[test]
fn wsl_host_clipboard_wins_over_installed_linux_backends() {
    assert_eq!(
        super::select_clipboard_backend(true, true, true, true),
        "windows_host_powershell"
    );
    assert_eq!(
        super::select_clipboard_backend(false, true, true, true),
        "wl-clipboard"
    );
}

#[cfg(unix)]
#[test]
fn file_clipboard_uri_percent_encodes_path_bytes() {
    let uri = super::file_uri(std::path::Path::new("/tmp/a file#1.txt")).expect("file URI");

    assert_eq!(uri, "file:///tmp/a%20file%231.txt");
}

#[test]
fn partial_download_detection_is_case_insensitive_and_bounded_to_suffixes() {
    assert!(super::is_partial_download_name("archive.zip.crdownload"));
    assert!(super::is_partial_download_name("video.PART"));
    assert!(!super::is_partial_download_name("part-notes.txt"));
    assert!(!super::is_partial_download_name("archive.zip"));
}

#[test]
fn download_inventory_distinguishes_complete_and_partial_files() {
    let directory =
        std::env::temp_dir().join(format!("agent-download-test-{}", super::now_unix_millis()));
    std::fs::create_dir(&directory).expect("create download test directory");
    std::fs::write(directory.join("ready.zip"), b"ready").expect("write complete file");
    std::fs::write(directory.join("pending.zip.part"), b"pending").expect("write partial file");

    let inventory = super::download_inventory_at(&directory, 10).expect("download inventory");
    std::fs::remove_dir_all(&directory).expect("remove download test directory");

    assert_eq!(inventory.get("count").and_then(Value::as_u64), Some(2));
    assert_eq!(
        inventory.get("partial_count").and_then(Value::as_u64),
        Some(1)
    );
}

#[test]
fn desktop_resolver_normalizes_and_scores_names() {
    let query = super::normalize_match_text("VS Code!");

    assert_eq!(query, "vs code");
    assert_eq!(super::match_score(&query, &["VS Code"]), 100);
    assert!(super::match_score(&query, &["Visual Studio Code"]) > 0);
    assert_eq!(super::match_score(&query, &["Terminal"]), 0);
}

#[test]
fn desktop_resolver_preserves_all_query_terms() {
    let query = super::normalize_match_text("my Spark app");

    assert_eq!(query, "my spark app");
    assert_eq!(super::match_score(&query, &["Spark"]), 13);
    assert_eq!(
        super::match_score(
            &super::normalize_match_text("Spark Uninstaller"),
            &["Spark"]
        ),
        20
    );
}

#[test]
fn windows_host_window_list_uses_real_window_handles() {
    let script = super::windows_host_window_list_script(10);

    assert!(script.contains("MainWindowHandle"));
    assert!(script.contains("'0x{0:x}'"));
    assert!(script.contains("Name='Pid'"));
}

#[test]
fn windows_host_focus_script_restores_and_focuses_handle() {
    let script = super::windows_host_focus_window_script("0x3a00007").expect("focus script");

    assert!(script.contains("ShowWindowAsync($hwnd, 9)"));
    assert!(script.contains("SetForegroundWindow($hwnd)"));
    assert!(super::windows_host_focus_window_script("0x3a00007;rm").is_err());
}

#[test]
fn launched_window_id_picks_new_window_after_launch() {
    let before = json!({"windows": {"ids": ["0x1", "0x2"]}});
    let after = json!({"windows": {"ids": ["0x1", "0x2", "0x3"]}});

    assert_eq!(
        super::launched_window_id(&before, &after).as_deref(),
        Some("0x3")
    );
    assert_eq!(super::launched_window_id(&after, &before), None);
}

#[test]
fn windows_shortcut_focus_query_uses_shortcut_name() {
    assert_eq!(
        super::windows_shortcut_focus_query(
            r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Spark.lnk"
        ),
        "Spark"
    );
}

#[test]
fn windows_host_close_script_requests_normal_window_close() {
    let script = super::windows_host_close_window_script("0x3a00007").expect("close script");

    assert!(script.contains("PostMessage($hwnd, 0x0010"));
    assert!(script.contains("IsWindow($hwnd)"));
    assert!(script.contains("$processId = 0"));
    assert!(script.contains("Stop-Process -Id $processId -Force"));
    assert!(!script.contains("$pid"));
    assert!(super::windows_host_close_window_script("0x3a00007;rm").is_err());
}

#[test]
fn windows_host_process_scripts_do_not_overwrite_builtin_pid_variable() {
    let terminate =
        super::windows_host_terminate_process_script(16608).expect("terminate process script");
    let state = super::windows_host_process_state_script(16608).expect("process state script");

    assert!(terminate.contains("$processId = 16608"));
    assert!(terminate.contains("Stop-Process -Id $processId -Force"));
    assert!(!terminate.contains("$pid"));
    assert!(state.contains("$processId = 16608"));
    assert!(state.contains("Get-Process -Id $processId"));
    assert!(!state.contains("$pid"));
    assert!(super::windows_host_terminate_process_script(1).is_err());
    assert!(super::windows_host_process_state_script(0).is_err());
}

#[test]
fn windows_shortcut_targets_are_identifier_safe_and_roundtrip() {
    let path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Spark.lnk";
    let target = super::windows_shortcut_target(path).expect("shortcut target");

    assert!(target.starts_with(super::WINDOWS_SHORTCUT_PREFIX));
    assert!(super::valid_identifier(&target));
    assert_eq!(
        super::decode_windows_shortcut_target(&target)
            .expect("decode")
            .as_deref(),
        Some(path)
    );
    assert!(super::windows_shortcut_target("C:\\Temp\\Spark.exe").is_none());
}

#[test]
fn windows_shortcut_launch_script_uses_file_path() {
    let script = super::windows_host_start_shortcut_script();

    assert!(script.contains("Test-Path -LiteralPath"));
    assert!(script.contains("Start-Process -FilePath"));
    assert!(script.contains(".lnk"));
}

#[test]
fn windows_registered_app_targets_are_identifier_safe_and_roundtrip() {
    let app_id = "Vitelglobal.Vitelglobal_v1ncde5y6f3mm!com.vitelglobal.vitelgloabalapp.winx";
    let target = super::windows_app_target(app_id).expect("app target");

    assert!(target.starts_with(super::WINDOWS_APP_PREFIX));
    assert!(super::valid_identifier(&target));
    assert_eq!(
        super::decode_windows_app_target(&target)
            .expect("decode")
            .as_deref(),
        Some(app_id)
    );
    assert!(super::windows_app_target("bad\nid").is_none());
}

#[test]
fn windows_registered_app_launch_script_uses_apps_folder() {
    let script = super::windows_host_start_app_script();

    assert!(script.contains("shell:AppsFolder"));
    assert!(script.contains("Start-Process"));
    assert!(script.contains("Invalid app id"));
}

#[test]
fn windows_registered_app_match_script_filters_before_limit() {
    let script = super::windows_host_registered_application_matches_script(10);

    assert!(script.contains("Where-Object"));
    assert!(script.contains("Contains($query)"));
    assert!(script.contains("Select-Object -First 10"));
}

#[test]
fn keyboard_key_specs_reject_shell_shapes() {
    assert_eq!(super::normalize_key_spec("ctrl+l").unwrap(), "ctrl+l");
    assert_eq!(super::normalize_key_spec("Alt+Tab").unwrap(), "Alt+Tab");
    assert!(super::normalize_key_spec("ctrl+l;rm").is_err());
    assert!(super::normalize_key_spec("").is_err());
}

#[test]
fn keyboard_type_receipt_redacts_text() {
    let receipt = super::keyboard_value_receipt("type_text", "secret typed text");

    assert_eq!(
        receipt.get("content_returned").and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(receipt.get("char_count").and_then(Value::as_u64), Some(17));
    assert!(receipt.get("secret typed text").is_none());
}

#[test]
fn pointer_coordinates_and_scroll_are_bounded() {
    assert_eq!(super::parse_pointer_coordinates("10,20").unwrap(), (10, 20));
    assert!(super::parse_pointer_coordinates("10;rm,20").is_err());
    assert!(super::parse_pointer_coordinates("-1,20").is_err());
    assert_eq!(super::pointer_button(Some("3")).unwrap(), 3);
    assert!(super::pointer_button(Some("4")).is_err());
    assert_eq!(super::parse_scroll_steps(Some("-5")).unwrap(), -5);
    assert!(super::parse_scroll_steps(Some("0")).is_err());
    assert!(super::parse_scroll_steps(Some("26")).is_err());
}

#[test]
fn windows_volume_script_uses_validated_scalar_and_verifies_result() {
    let script = windows_host_volume_script(0);

    assert!(script.contains("[AgentCoreAudio]::SetVolume(0)"));
    assert!(script.contains("after_percent"));
    assert!(script.contains("ConvertTo-Json -Compress"));
    assert!(script.contains("5CDF2C82-841E-4546-9722-0CF74078229A"));
}

#[test]
fn windows_volume_script_converts_percent_to_scalar() {
    let script = windows_host_volume_script(75);

    assert!(script.contains("[AgentCoreAudio]::SetVolume(0.75)"));
}
