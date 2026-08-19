use anyhow::Result;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Debug)]
pub(super) struct CommandOutput {
    pub(super) status: i32,
    pub(super) stdout: String,
    pub(super) stderr: String,
}

pub(super) fn required_target<'a>(command: &str, target: Option<&'a str>) -> Result<&'a str> {
    target.ok_or_else(|| anyhow::anyhow!("{} requires --target", command))
}

pub(super) fn run_capture<const N: usize>(argv: &[&str; N]) -> Result<CommandOutput> {
    run_capture_dynamic(argv)
}

pub(super) fn run_capture_dynamic<T: AsRef<OsStr>>(argv: &[T]) -> Result<CommandOutput> {
    let Some((program, arguments)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty command"));
    };
    let output = Command::new(program.as_ref())
        .args(arguments.iter().map(AsRef::as_ref))
        .output()?;
    Ok(command_output(output))
}

pub(super) fn run_capture_with_stdin(argv: &[&str], input: &str) -> Result<CommandOutput> {
    let Some((program, arguments)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty command"));
    };
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input.as_bytes())?;
    }
    Ok(command_output(child.wait_with_output()?))
}

pub(super) fn read_trimmed(path: PathBuf) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn command_output(output: std::process::Output) -> CommandOutput {
    CommandOutput {
        status: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    }
}
