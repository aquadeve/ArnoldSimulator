#pragma once

#include <memory>
#include <string>

#include "brain.h"
#include "components.h"
#include "luau_engine.h"

namespace PicoLlmModel
{

class PicoLlmBrain : public Brain
{
public:
    static const char* Type;

    PicoLlmBrain(BrainBase& base, json& params);
    virtual ~PicoLlmBrain();

    virtual void pup(PUP::er& p) override;
    virtual const char* GetType() const override;
    virtual void Control(size_t brainStep) override;
    virtual void AcceptContributionFromRegion(RegionIndex regIdx, const uint8_t* contribution, size_t size) override;

private:
    void Load();
    std::string ResolvePrompt(size_t brainStep) const;
    std::string Generate(const std::string& prompt);

    std::string mModelPath;
    std::string mInitialPrompt;
    std::string mLuauScript;
    std::string mLastResponse;

    size_t mMaxResponseTokens;
    size_t mBrainStepsPerResponse;
    float mTemperature;
    float mTopP;
    uint64_t mSeed;
    int mThreads;

    bool mLoaded;

    std::unique_ptr<LuauEngine> mScriptEngine;

    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

void Init(NeuronFactory* neuronFactory, RegionFactory* regionFactory, BrainFactory* brainFactory);

}
