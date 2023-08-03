/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

enum FFTOrder
{
    order2048 = 11,
    order4096 = 12,
    order8192 = 13
};

template<typename BlockType>
struct FFTDataGenerator
{
    /* Produce the FFT data from an audio buffer */
    void productFFTDataForRendering(const juce::AudioBuffer<float>& audioData, const float negativeInfinity)
    {
        const auto fftSize = getFFTSize();

        fftData.assign(fftData.size(), 0);
        auto* readIndex = audioData.getReadPointer(0);
        std::copy(readIndex, readIndex + fftSize, fftData.begin());

        // apply a windowing function to data
        window->multiplyWithWindowingTable(fftData.data(), fftSize);

        // render FFT data
        forwardFFT->performFrequencyOnlyForwardTransform(fftData.data());

        int numBins = (int)fftSize / 2;

        // normalize fft values
        for (int i = 0; i < numBins; ++i)
        {
            fftData[i] /= (float)numBins;
        }

        // convert to decibels
        for (int i = 0; i < numBins; ++i)
        {
            fftData[i] = juce::Decibels::gainToDecibels(fftData[i], negativeInfinity);
        }

        fftDataFifo.push(fftData);
    }

    void changeOrder(FFTOrder newOrder)
    {
        // when you change order, recreate the window, forward FFT, fifo, fftData
        // also reset the fifoindex
        // things that need recreating should be created on the heap via std::make_unique<>

        order = newOrder;
        auto fftSize = getFFTSize();

        forwardFFT = std::make_unique<juce::dsp::FFT>(order);
        window = std::make_unique<juce::dsp::WindowingFunction<float>>(fftSize, juce::dsp::WindowingFunction<float>::blackmanHarris);

        fftData.clear();
        fftData.resize(fftSize * 2, 0);
        fftDataFifo.prepare(fftData.size());
    }

    int getFFTSize() const { return 1 << order; }
    int getNumAvailableFFTDataBlocks() const { return fftDataFifo.getNumAvailableForReading(); }

    bool getFFTData(BlockType& fftData) { return fftDataFifo.pull(fftData); }

private:
    FFTOrder order;
    BlockType fftData;
    std::unique_ptr<juce::dsp::FFT> forwardFFT;
    std::unique_ptr<juce::dsp::WindowingFunction<float>> window;

    Fifo<BlockType> fftDataFifo;
};

template<typename PathType>
struct AnalyzedPathGenerator
{
    /* converts 'renderData[] into a juce::path*/
    void generatePath(const std::vector<float>& renderData,
        juce::Rectangle<float> fftBounds,
        int fftSize,
        float binWidth,
        float negativeInfinity)
    {
        auto top = fftBounds.getY();
        auto bottom = fftBounds.getHeight();
        auto width = fftBounds.getWidth();

        int numBins = (int)fftSize / 2;

        PathType p;
        p.preallocateSpace(3 * (int)fftBounds.getWidth());

        auto map = [bottom, top, negativeInfinity](float y)
        {
            return juce::jmap(y,
                negativeInfinity, 0.f,
                float(bottom), top);
        };

        auto y = map(renderData[0]);

        jassert(!std::isnan(y) && !std::isinf(y));

        p.startNewSubPath(0, y);

        const int pathResolution = 2; //you can draw line-to's every pathResolution pixels

        for (int binNum = 1; binNum < numBins; binNum += pathResolution)
        {
            y = map(renderData[binNum]);
            jassert(!std::isnan(y) && !std::isinf(y));

            if (!std::isnan(y) && !std::isinf(y))
            {
                auto binFreq = binNum * binWidth;
                auto normalizedBinX = juce::mapFromLog10(binFreq, 20.f, 20000.f);
                int binX = std::floor(normalizedBinX * width);
                p.lineTo(binX, y);
            }
        }

        pathFifo.push(p);
    }

    int getNumPathAvailable() const
    {
        return pathFifo.getNumAvailableForReading();
    }

    bool getPath(PathType& path)
    {
        return pathFifo.pull(path);
    }
private:
    Fifo<PathType> pathFifo;
};


struct LookAndFeel : juce::LookAndFeel_V4
{
    void drawRotarySlider(juce::Graphics&,
        int x, int y, int width, int height,
        float sliderPosProportional,
        float rotaryStartAngle,
        float rotaryEndAngle,
        juce::Slider&) override;

    void drawToggleButton(juce::Graphics& g,
        juce::ToggleButton& toggleButton,
        bool shouldDrawButtonAsHighlighted,
        bool shouldDrawButtonAsDown) override;
};

struct RotarySliderWithLabels : juce::Slider
{
    RotarySliderWithLabels(juce::RangedAudioParameter& rap, const juce::String& unitSuffix) : juce::Slider(juce::Slider::SliderStyle::RotaryHorizontalVerticalDrag,
        juce::Slider::TextEntryBoxPosition::NoTextBox),
        param(&rap),
        suffix(unitSuffix)
    {
        setLookAndFeel(&lnf);
    }

    ~RotarySliderWithLabels()
    {
        setLookAndFeel(nullptr);
    }

    struct LabelPos
    {
        float pos;
        juce::String label;
    };

    juce::Array<LabelPos> labels;

    void paint(juce::Graphics& g) override;
    juce::Rectangle<int> getSliderBounds() const;
    int getTextHeight() const { return 14; }
    juce::String getDisplayString() const;

private:
    LookAndFeel lnf;

    juce::RangedAudioParameter* param;
    juce::String suffix;
};

struct PathProducer
{
    PathProducer(SingleChannelSampleFifo<SimpleEQ22AudioProcessor::BlockType>& scsf) :
        leftChannelFifo(&scsf)
    {
        leftChannelFFTDataGenerator.changeOrder(FFTOrder::order2048);
        monoBuffer.setSize(1, leftChannelFFTDataGenerator.getFFTSize());

    }

    void process(juce::Rectangle<float> fftBounds, double sampleRate);
    juce::Path getPath() { return leftChannelFFTPath; }

private:
    SingleChannelSampleFifo<SimpleEQ22AudioProcessor::BlockType>* leftChannelFifo;

    juce::AudioBuffer<float> monoBuffer;

    FFTDataGenerator<std::vector<float>> leftChannelFFTDataGenerator;

    AnalyzedPathGenerator<juce::Path> pathProducer;

    juce::Path leftChannelFFTPath;
};

struct ResponseCurveComponent : juce::Component,
    juce::AudioProcessorParameter::Listener,
    juce::Timer
{
    ResponseCurveComponent(SimpleEQ22AudioProcessor&);
    ~ResponseCurveComponent();

    void parameterValueChanged(int parameterIndex, float newValue) override;
    void parameterGestureChanged(int parameterIndex, bool gestureIsStarting) override {}
    void timerCallback() override;
    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    SimpleEQ22AudioProcessor& audioProcessor;
    juce::Atomic<bool> parametersChanged {false};

    MonoChain monoChain;

    void updateChain();

    juce::Image background;
    juce::Rectangle<int> getRenderArea();
    juce::Rectangle<int> getAnalysisArea();

    PathProducer leftPathProducer, rightPathProducer;
};

//==============================================================================
/**
*/
class SimpleEQ22AudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    SimpleEQ22AudioProcessorEditor (SimpleEQ22AudioProcessor&);
    ~SimpleEQ22AudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    SimpleEQ22AudioProcessor& audioProcessor;

    RotarySliderWithLabels peakFreqSlider, 
        peakGainSlider, 
        peakQualitySlider, 
        lowCutFreqSlider, 
        highCutFreqSlider, 
        lowCutSlopeSlider, 
        highCutSlopeSlider;

    ResponseCurveComponent responseCurveComponent;

    using APVTS = juce::AudioProcessorValueTreeState;
    using Attachment = APVTS::SliderAttachment;
    Attachment peakFreqSliderAttachment, 
        peakGainSliderAttachment, 
        peakQualitySliderAttachment, 
        lowCutFreqSliderAttachment, 
        highCutFreqSliderAttachment, 
        lowCutSlopeSliderAttachment, 
        highCutSlopeSliderAttachment;

    juce::ToggleButton lowCutBypassButton, 
        peakBypassButton, 
        highCutBypassButton, 
        analyzerEnabledButton;

    using ButtonAttachment = APVTS::ButtonAttachment;
    ButtonAttachment lowCutBypassButtonAttachment, 
        peakBypassButtonAttachment,
        highCutBypassButtonAttachment,
        analyzerEnabledButtonAttachment;

    std::vector<juce::Component*> getComps();

    LookAndFeel lnf;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SimpleEQ22AudioProcessorEditor)
};
