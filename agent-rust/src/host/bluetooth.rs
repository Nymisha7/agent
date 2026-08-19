pub(super) fn info_value(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (name, value) = line.trim().split_once(':')?;
        (name == key).then(|| value.trim().to_string())
    })
}
