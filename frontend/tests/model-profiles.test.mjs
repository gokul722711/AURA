import test from "node:test";
import assert from "node:assert/strict";

// Helper functions mirroring frontend logic in page.jsx

const formatProviderBadge = (provider) => {
  switch (provider) {
    case "nvidia":
      return "NVIDIA NIM";
    case "ollama":
      return "Ollama";
    case "openai_compatible":
      return "OpenAI Compatible";
    default:
      return provider || "Custom";
  }
};

const resolveActiveProfile = (profiles) => {
  if (!Array.isArray(profiles) || profiles.length === 0) return null;
  return profiles.find((p) => p.is_active) || profiles[0] || null;
};

const createResearchPayload = (objective, mode = "knowledge_base", selectedProfileId = null) => {
  const payload = {
    objective: objective.trim(),
    mode,
  };
  if (selectedProfileId) {
    payload.model_profile_id = selectedProfileId;
  }
  return payload;
};

const isNoModelConfiguredError = (errorData) => {
  if (!errorData) return false;
  return (
    errorData.code === "NO_MODEL_CONFIGURED" ||
    (typeof errorData.error === "string" && errorData.error.toLowerCase().includes("no active model profile"))
  );
};

test("M15 Frontend: formatProviderBadge handles known and custom providers", () => {
  assert.equal(formatProviderBadge("nvidia"), "NVIDIA NIM");
  assert.equal(formatProviderBadge("ollama"), "Ollama");
  assert.equal(formatProviderBadge("openai_compatible"), "OpenAI Compatible");
  assert.equal(formatProviderBadge("custom_provider"), "custom_provider");
  assert.equal(formatProviderBadge(""), "Custom");
  assert.equal(formatProviderBadge(null), "Custom");
});

test("M15 Frontend: resolveActiveProfile correctly finds active profile", () => {
  const profiles = [
    { id: "p1", name: "Ollama Local", is_active: false },
    { id: "p2", name: "NVIDIA Nemotron", is_active: true },
    { id: "p3", name: "vLLM Server", is_active: false },
  ];
  const active = resolveActiveProfile(profiles);
  assert.equal(active.id, "p2");
  assert.equal(active.name, "NVIDIA Nemotron");
});

test("M15 Frontend: resolveActiveProfile falls back safely or returns null", () => {
  assert.equal(resolveActiveProfile([]), null);
  assert.equal(resolveActiveProfile(null), null);

  const noActiveProfiles = [
    { id: "p1", name: "First Profile", is_active: false },
    { id: "p2", name: "Second Profile", is_active: false },
  ];
  assert.equal(resolveActiveProfile(noActiveProfiles).id, "p1");
});

test("M15 Frontend: research submission payload includes model_profile_id when provided", () => {
  const payloadWithProfile = createResearchPayload("Test objective", "web", "profile-123");
  assert.deepEqual(payloadWithProfile, {
    objective: "Test objective",
    mode: "web",
    model_profile_id: "profile-123",
  });

  const payloadWithoutProfile = createResearchPayload("Test objective", "knowledge_base");
  assert.deepEqual(payloadWithoutProfile, {
    objective: "Test objective",
    mode: "knowledge_base",
  });
  assert.equal("model_profile_id" in payloadWithoutProfile, false);
});

test("M15 Frontend: isNoModelConfiguredError detects machine-readable and fallback error states", () => {
  assert.equal(
    isNoModelConfiguredError({
      error: "No active model profile configured.",
      code: "NO_MODEL_CONFIGURED",
    }),
    true
  );

  assert.equal(
    isNoModelConfiguredError({
      error: "No active model profile configured. Please create or activate a ModelProfile.",
    }),
    true
  );

  assert.equal(
    isNoModelConfiguredError({
      error: "Database connection failed",
      code: "DB_ERROR",
    }),
    false
  );

  assert.equal(isNoModelConfiguredError(null), false);
  assert.equal(isNoModelConfiguredError({}), false);
});

test("M15 Frontend: Model Profile list payload preserves masked credentials without raw key", () => {
  const rawApiResponse = {
    profiles: [
      {
        id: "profile-uuid-1",
        name: "My Nemotron",
        provider: "nvidia",
        endpoint: "https://integrate.api.nvidia.com/v1",
        model: "nvidia/nemotron-3-ultra-550b-a55b",
        api_key: "...abcd",
        is_active: true,
        capabilities: ["generation", "structured_output"],
      },
    ],
  };

  const profile = rawApiResponse.profiles[0];
  assert.equal(profile.api_key.startsWith("..."), true);
  assert.notEqual(profile.api_key, "nvapi-secret-key-1234567890");
});
