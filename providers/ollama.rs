// crates/nym-core/src/providers/types.rs

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelProvider {
    pub id: String,
    pub name: String,
    pub provider_type: ProviderType,
    pub base_url: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderType {
    Ollama,
    OpenAICompatible,
}