using Microsoft.ML.OnnxRuntimeGenAI;
using Qwen3.Onnx.Utils;

namespace Qwen3.Onnx.Llm.Tests;

public class LlmSanityTests : IDisposable
{
    private readonly Model _model;
    private readonly Tokenizer _tokenizer;

    public LlmSanityTests()
    {
        var modelPath = RepositoryPaths.GetLlmModelPath();
        _model = new Model(modelPath);
        _tokenizer = new Tokenizer(_model);
    }

    [Theory]
    [InlineData("What is 2+2?", "4")]
    [InlineData("What is the capital of France?", "Paris")]
    public void Generate_ShouldProduceRelevantResponse(string prompt, string expectedContent)
    {
        var formattedPrompt = $"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";

        using var inputTokens = _tokenizer.Encode(formattedPrompt);
        using var generatorParams = new GeneratorParams(_model);
        generatorParams.SetSearchOption("max_length", 100);
        generatorParams.SetSearchOption("temperature", 0.6);

        using var generator = new Generator(_model, generatorParams);
        generator.AppendTokens(inputTokens[0]);

        var outputTokens = new List<int>();
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
            var newToken = generator.GetSequence(0)[^1];
            outputTokens.Add(newToken);
        }

        var response = _tokenizer.Decode([.. outputTokens]);

        Assert.NotNull(response);
        Assert.NotEmpty(response);

        if (!string.IsNullOrEmpty(expectedContent))
        {
            Assert.Contains(expectedContent, response, StringComparison.OrdinalIgnoreCase);
        }
    }

    public void Dispose()
    {
        _tokenizer?.Dispose();
        _model?.Dispose();

        GC.SuppressFinalize(this);
    }
}
