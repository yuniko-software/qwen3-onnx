using Microsoft.ML.OnnxRuntimeGenAI;
using Qwen3.Onnx.Utils;

var modelPath = RepositoryPaths.GetLlmModelPath();

Console.WriteLine("Loading model...");

using var model = new Model(modelPath);
Console.WriteLine("Model loaded");

using var tokenizer = new Tokenizer(model);
using var tokenizerStream = tokenizer.CreateStream();
Console.WriteLine("Tokenizer created\n");

while (true)
{
    Console.Write("Prompt (Use quit() to exit): ");
    var text = Console.ReadLine();

    if (string.IsNullOrEmpty(text))
    {
        Console.WriteLine("Error, input cannot be empty");
        continue;
    }

    if (string.Equals(text, "quit()", StringComparison.Ordinal))
    {
        break;
    }

    var formattedPrompt = $"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n";

    using var inputTokens = tokenizer.Encode(formattedPrompt);

    using var generatorParams = new GeneratorParams(model);
    generatorParams.SetSearchOption("max_length", 2048);
    generatorParams.SetSearchOption("temperature", 0.6);
    generatorParams.SetSearchOption("top_p", 0.95);
    generatorParams.SetSearchOption("top_k", 20);

    using var generator = new Generator(model, generatorParams);
    generator.AppendTokens(inputTokens[0]);

    Console.Write("\nOutput: ");

    try
    {
        while (!generator.IsDone())
        {
            generator.GenerateNextToken();
            var newToken = generator.GetSequence(0)[^1];
            Console.Write(tokenizerStream.Decode(newToken));
        }
    }
    catch (Exception)
    {
        Console.WriteLine("  --control+c pressed, aborting generation--");
    }

    Console.WriteLine("\n");
}
