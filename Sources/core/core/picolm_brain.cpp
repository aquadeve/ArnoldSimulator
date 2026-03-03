#include "picolm_brain.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

extern "C"
{
#include "../../../picolm/model.h"
#include "../../../picolm/tokenizer.h"
#include "../../../picolm/sampler.h"
#include "../../../picolm/grammar.h"
#include "../../../picolm/tensor.h"

#include "../../../picolm/quant.c"
#include "../../../picolm/tensor.c"
#include "../../../picolm/model.c"
#include "../../../picolm/tokenizer.c"
#include "../../../picolm/sampler.c"
#include "../../../picolm/grammar.c"
}

extern "C"
{
#include "../../ThirdParty/luau/VM/include/lua.h"
#include "../../ThirdParty/luau/VM/include/lauxlib.h"
}

#include "log.h"

namespace PicoLlmModel
{

struct PicoLlmBrain::Impl
{
    model_t model;
    tokenizer_t tokenizer;
    sampler_t sampler;
    grammar_state_t grammar;

    Impl()
    {
        memset(&model, 0, sizeof(model));
        memset(&tokenizer, 0, sizeof(tokenizer));
        memset(&sampler, 0, sizeof(sampler));
        memset(&grammar, 0, sizeof(grammar));
    }
};

namespace
{
const char* ReadString(const json& params, const char* key, const char* fallback = "")
{
    if (params.find(key) != params.end() && params[key].is_string()) {
        return params[key].get_ref<const std::string&>().c_str();
    }

    return fallback;
}

template <typename T>
T ReadNumber(const json& params, const char* key, T fallback)
{
    if (params.find(key) != params.end()) {
        return params[key].get<T>();
    }

    return fallback;
}
}

const char* PicoLlmBrain::Type = "PicoLlmBrain";

PicoLlmBrain::PicoLlmBrain(BrainBase& base, json& params) :
    Brain(base, params),
    mModelPath(ReadString(params, "modelPath")),
    mInitialPrompt(ReadString(params, "initialPrompt", "You are a thoughtful and concise assistant.")),
    mLuauScript(ReadString(params, "script")),
    mLastResponse(),
    mMaxResponseTokens(ReadNumber<size_t>(params, "maxResponseTokens", 64)),
    mBrainStepsPerResponse(ReadNumber<size_t>(params, "brainStepsPerResponse", 10)),
    mTemperature(ReadNumber<float>(params, "temperature", 0.8f)),
    mTopP(ReadNumber<float>(params, "topP", 0.9f)),
    mSeed(ReadNumber<uint64_t>(params, "seed", 42)),
    mThreads(ReadNumber<int>(params, "threads", 2)),
    mLoaded(false),
    mScriptEngine(),
    mImpl(new Impl())
{
    if (!mLuauScript.empty()) {
        mScriptEngine.reset(new LuauEngine());
        mScriptEngine->Load(mLuauScript, "PicoLlmBrainScript");
    }

    if (!mModelPath.empty()) {
        Load();
    } else {
        Log(LogLevel::Warning, "[PicoLlmBrain] Parameter 'modelPath' missing; brain will stay idle.");
    }
}

PicoLlmBrain::~PicoLlmBrain()
{
    if (mLoaded) {
        grammar_free(&mImpl->grammar);
        tokenizer_free(&mImpl->tokenizer);
        model_free(&mImpl->model);
    }
}

void PicoLlmBrain::pup(PUP::er& p)
{
    p | mModelPath;
    p | mInitialPrompt;
    p | mLuauScript;
    p | mLastResponse;
    p | mMaxResponseTokens;
    p | mBrainStepsPerResponse;
    p | mTemperature;
    p | mTopP;
    p | mSeed;
    p | mThreads;
    p | mLoaded;

    if (p.isUnpacking()) {
        mImpl.reset(new Impl());

        if (!mLuauScript.empty()) {
            mScriptEngine.reset(new LuauEngine());
            mScriptEngine->Load(mLuauScript, "PicoLlmBrainScript");
        }

        if (!mModelPath.empty()) {
            Load();
        }
    }
}

const char* PicoLlmBrain::GetType() const
{
    return Type;
}

void PicoLlmBrain::Load()
{
    if (model_load(&mImpl->model, mModelPath.c_str(), 0) != 0) {
        throw std::runtime_error(std::string("[PicoLlmBrain] Failed to load model: ") + mModelPath);
    }

    if (tokenizer_load(&mImpl->tokenizer, &mImpl->model) != 0) {
        model_free(&mImpl->model);
        throw std::runtime_error("[PicoLlmBrain] Failed to load tokenizer.");
    }

    sampler_init(&mImpl->sampler, mTemperature, mTopP, mSeed);
    grammar_init(&mImpl->grammar, GRAMMAR_NONE, &mImpl->tokenizer);

    tensor_set_threads(std::max(1, mThreads));
    mLoaded = true;

    Log(LogLevel::Info, "[PicoLlmBrain] Loaded model '%s'", mModelPath.c_str());
}

std::string PicoLlmBrain::ResolvePrompt(size_t brainStep) const
{
    if (!mScriptEngine || !mScriptEngine->HasFunction("build_prompt")) {
        return mInitialPrompt;
    }

    lua_State* state = mScriptEngine->GetState();
    lua_pushnumber(state, static_cast<double>(brainStep));
    lua_pushstring(state, mLastResponse.c_str());
    mScriptEngine->Call("build_prompt", 2, 1);

    const char* result = luaL_optstring(state, -1, mInitialPrompt.c_str());
    std::string prompt = result ? result : mInitialPrompt;
    lua_pop(state, 1);

    return prompt;
}

std::string PicoLlmBrain::Generate(const std::string& prompt)
{
    grammar_free(&mImpl->grammar);
    grammar_init(&mImpl->grammar, GRAMMAR_NONE, &mImpl->tokenizer);

    std::vector<int> promptTokens(prompt.size() + 8, 0);
    const int nPrompt = tokenizer_encode(&mImpl->tokenizer, prompt.c_str(), promptTokens.data(), static_cast<int>(promptTokens.size()), 1);

    if (nPrompt <= 0) {
        return "";
    }

    std::ostringstream out;
    int token = promptTokens[0];
    int maxPos = std::min(mImpl->model.config.max_seq_len, nPrompt + static_cast<int>(mMaxResponseTokens));

    for (int pos = 0; pos < maxPos; ++pos) {
        float* logits = model_forward(&mImpl->model, token, pos);

        int nextToken = 0;
        if (pos < nPrompt - 1) {
            nextToken = promptTokens[pos + 1];
        } else {
            grammar_apply(&mImpl->grammar, logits, mImpl->model.config.vocab_size);
            nextToken = sampler_sample(&mImpl->sampler, logits, mImpl->model.config.vocab_size);
            grammar_advance(&mImpl->grammar, &mImpl->tokenizer, nextToken);

            const char* piece = tokenizer_decode(&mImpl->tokenizer, token, nextToken);
            if (piece) {
                out << piece;
            }

            if (nextToken == static_cast<int>(mImpl->tokenizer.eos_id) || grammar_is_complete(&mImpl->grammar)) {
                break;
            }
        }

        token = nextToken;
    }

    return out.str();
}

void PicoLlmBrain::Control(size_t brainStep)
{
    if (!mLoaded || mBrainStepsPerResponse == 0 || brainStep % mBrainStepsPerResponse != 0) {
        return;
    }

    const std::string prompt = ResolvePrompt(brainStep);
    mLastResponse = Generate(prompt);

    if (!mLastResponse.empty()) {
        Log(LogLevel::Info, "[PicoLlmBrain][%zu] %s", brainStep, mLastResponse.c_str());
    }
}

void PicoLlmBrain::AcceptContributionFromRegion(RegionIndex regIdx, const uint8_t* contribution, size_t size)
{
    (void)regIdx;
    (void)contribution;
    (void)size;
}

void Init(NeuronFactory* neuronFactory, RegionFactory* regionFactory, BrainFactory* brainFactory)
{
    (void)neuronFactory;
    (void)regionFactory;
    brainFactory->Register(PicoLlmBrain::Type, BrainBuilder<PicoLlmBrain>);
}

}
