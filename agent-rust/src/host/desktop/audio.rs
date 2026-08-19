use super::*;

pub(super) fn windows_host_set_volume(
    percent: u8,
    target: Option<&str>,
    value: Option<&str>,
) -> Result<Value> {
    let script = windows_host_volume_script(percent);
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])?;
    let state = output
        .stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<Value>(line.trim()).ok());
    let after_percent = state
        .as_ref()
        .and_then(|item| item.get("after_percent"))
        .and_then(Value::as_u64);
    let verified = output.status == 0 && after_percent == Some(u64::from(percent));

    Ok(json!({
        "ok": verified,
        "tool": "desktop_action",
        "action": "set_volume",
        "target": target,
        "value": value,
        "runtime": "windows_host",
        "exit_code": output.status,
        "before": state.as_ref().and_then(|item| item.get("before_percent")).cloned(),
        "after": after_percent,
        "verified": verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "reason": if output.status != 0 { Some("command_failed") } else if !verified { Some("verification_failed") } else { None },
        "stdout": if state.is_none() { Some(output.stdout) } else { None },
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_volume_script(percent: u8) -> String {
    const CORE_AUDIO_TYPE: &str = r#"using System;
using System.Runtime.InteropServices;

public static class AgentCoreAudio {
    [ComImport]
    [Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
    private class MMDeviceEnumeratorComObject { }

    private enum EDataFlow { Render, Capture, All }
    private enum ERole { Console, Multimedia, Communications }

    [ComImport]
    [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMMDeviceEnumerator {
        int EnumAudioEndpoints(EDataFlow dataFlow, int stateMask, out IntPtr devices);
        [PreserveSig]
        int GetDefaultAudioEndpoint(EDataFlow dataFlow, ERole role, out IMMDevice device);
    }

    [ComImport]
    [Guid("D666063F-1587-4E43-81F1-B948E807363F")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMMDevice {
        [PreserveSig]
        int Activate(ref Guid iid, int classContext, IntPtr activationParameters,
            [MarshalAs(UnmanagedType.IUnknown)] out object instance);
    }

    [ComImport]
    [Guid("5CDF2C82-841E-4546-9722-0CF74078229A")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAudioEndpointVolume {
        int RegisterControlChangeNotify(IntPtr notify);
        int UnregisterControlChangeNotify(IntPtr notify);
        int GetChannelCount(ref uint channelCount);
        int SetMasterVolumeLevel(float levelDb, Guid eventContext);
        int SetMasterVolumeLevelScalar(float level, Guid eventContext);
        int GetMasterVolumeLevel(ref float levelDb);
        int GetMasterVolumeLevelScalar(ref float level);
    }

    private static IAudioEndpointVolume Endpoint() {
        var enumerator = (IMMDeviceEnumerator)new MMDeviceEnumeratorComObject();
        IMMDevice device;
        Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(
            EDataFlow.Render, ERole.Multimedia, out device));
        Guid iid = typeof(IAudioEndpointVolume).GUID;
        object endpoint;
        Marshal.ThrowExceptionForHR(device.Activate(ref iid, 23, IntPtr.Zero, out endpoint));
        return (IAudioEndpointVolume)endpoint;
    }

    public static float GetVolume() {
        float value = 0;
        Marshal.ThrowExceptionForHR(Endpoint().GetMasterVolumeLevelScalar(ref value));
        return value;
    }

    public static void SetVolume(float value) {
        Marshal.ThrowExceptionForHR(Endpoint().SetMasterVolumeLevelScalar(value, Guid.Empty));
    }
}"#;

    let scalar = f32::from(percent) / 100.0;
    format!(
        "$ErrorActionPreference = 'Stop'\nAdd-Type -TypeDefinition @'\n{}\n'@\n$before = [AgentCoreAudio]::GetVolume()\n[AgentCoreAudio]::SetVolume({scalar})\nStart-Sleep -Milliseconds 100\n$after = [AgentCoreAudio]::GetVolume()\n@{{before_percent = [math]::Round($before * 100); after_percent = [math]::Round($after * 100)}} | ConvertTo-Json -Compress",
        CORE_AUDIO_TYPE
    )
}

pub(super) fn audio_observation() -> Result<Value> {
    if !command_exists("pactl") {
        return Ok(json!({
            "ok": false,
            "backend": "unavailable",
            "reason": "backend_unavailable",
            "error": "Audio observation requires pactl.",
        }));
    }

    let volume = run_capture_dynamic(&["pactl", "get-sink-volume", "@DEFAULT_SINK@"])?;
    let mute = run_capture_dynamic(&["pactl", "get-sink-mute", "@DEFAULT_SINK@"])?;
    let sink = run_capture_dynamic(&["pactl", "get-default-sink"])?;
    let volume_percent = parse_pactl_volume_percent(&volume.stdout);
    let muted = parse_pactl_mute(&mute.stdout);
    let ok = volume.status == 0
        && mute.status == 0
        && sink.status == 0
        && volume_percent.is_some()
        && muted.is_some();
    let mut stderr = String::new();
    for value in [volume.stderr, mute.stderr, sink.stderr] {
        if value.is_empty() {
            continue;
        }
        if !stderr.is_empty() {
            stderr.push('\n');
        }
        stderr.push_str(&value);
    }
    Ok(json!({
        "ok": ok,
        "backend": "pactl",
        "sink": if sink.stdout.is_empty() { None } else { Some(sink.stdout) },
        "volume_percent": volume_percent,
        "muted": muted,
        "reason": if ok { None } else { Some("observation_failed") },
        "stderr": stderr,
    }))
}

pub(super) fn parse_pactl_volume_percent(output: &str) -> Option<u8> {
    output.split_whitespace().find_map(|field| {
        field
            .strip_suffix('%')
            .and_then(|value| value.parse::<u8>().ok())
    })
}

pub(super) fn parse_pactl_mute(output: &str) -> Option<bool> {
    match output
        .split_whitespace()
        .last()?
        .to_ascii_lowercase()
        .as_str()
    {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

pub(super) fn audio_state(kind: &str) -> Value {
    let command = if kind == "mute" {
        "get-sink-mute"
    } else {
        "get-sink-volume"
    };
    run_capture_dynamic(&["pactl", command, "@DEFAULT_SINK@"])
        .map(|output| json!({ "available": true, "value": output.stdout }))
        .unwrap_or_else(|_| json!({ "available": false }))
}
