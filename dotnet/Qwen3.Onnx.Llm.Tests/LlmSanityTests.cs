using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;
using Qwen3.Onnx.Utils;

namespace Qwen3.Onnx.Llm.Tests;

public sealed class LlmSanityTests : IDisposable
{
    private readonly Model _cpuModel;
    private readonly Tokenizer _cpuTokenizer;
    private readonly Model? _cudaModel;
    private readonly Tokenizer? _cudaTokenizer;
    private readonly bool _cudaAvailable;

    public LlmSanityTests()
    {
        var modelPath = RepositoryPaths.GetLlmModelPath();

        _cpuModel = new Model(modelPath);
        _cpuTokenizer = new Tokenizer(_cpuModel);

        try
        {
            _cudaModel = new Model(modelPath);
            _cudaTokenizer = new Tokenizer(_cudaModel);
            _cudaAvailable = true;
        }
        catch (OnnxRuntimeException)
        {
            _cudaModel = null;
            _cudaTokenizer = null;
            _cudaAvailable = false;
        }
    }

    [Theory]
    [InlineData("What is 2+2?", "4")]
    [InlineData("What is the capital of France?", "Paris")]
    public void CpuGenerate_ShouldProduceRelevantResponse(string prompt, string expectedContent)
    {
        ValidateGenerationQuality(_cpuModel, _cpuTokenizer, prompt, expectedContent);
    }

    [SkippableTheory]
    [InlineData("What is 2+2?", "4")]
    [InlineData("What is the capital of France?", "Paris")]
    public void CudaGenerate_ShouldProduceRelevantResponse(string prompt, string expectedContent)
    {
        Skip.If(!_cudaAvailable, "CUDA is not available on this system");

        Assert.NotNull(_cudaModel);
        Assert.NotNull(_cudaTokenizer);
        ValidateGenerationQuality(_cudaModel, _cudaTokenizer, prompt, expectedContent);
    }

    private static void ValidateGenerationQuality(Model model, Tokenizer tokenizer, string prompt, string expectedContent)
    {
        var formattedPrompt = $"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";

        using var inputTokens = tokenizer.Encode(formattedPrompt);
        using var generatorParams = new GeneratorParams(model);
        generatorParams.SetSearchOption("temperature", 0.0);
        generatorParams.SetSearchOption("top_k", 1);

        using var generator = new Generator(model, generatorParams);
        generator.AppendTokens(inputTokens[0]);

        var outputTokens = new List<int>();
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
            var newToken = generator.GetSequence(0)[^1];
            outputTokens.Add(newToken);
        }

        var response = tokenizer.Decode([.. outputTokens]);

        Assert.NotNull(response);
        Assert.NotEmpty(response);

        if (!string.IsNullOrEmpty(expectedContent))
        {
            Assert.Contains(expectedContent, response, StringComparison.OrdinalIgnoreCase);
        }
    }

    public void Dispose()
    {
        _cudaTokenizer?.Dispose();
        _cudaModel?.Dispose();
        _cpuTokenizer?.Dispose();
        _cpuModel?.Dispose();

        GC.SuppressFinalize(this);
    }
}
