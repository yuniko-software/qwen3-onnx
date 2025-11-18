namespace Qwen3.Onnx.Utils;

public static class RepositoryPaths
{
    public static string GetLlmModelPath()
    {
        return Path.Combine(FindRepositoryRoot(), "models", "qwen3-llm", "model");
    }

    public static string GetEmbeddingModelPath()
    {
        return Path.Combine(FindRepositoryRoot(), "models", "qwen3-embedding", "model", "model.onnx");
    }

    public static string FindRepositoryRoot()
    {
        var currentDir = new DirectoryInfo(Directory.GetCurrentDirectory());

        for (int i = 0; i < 10; i++)
        {
            var modelsDir = Path.Combine(currentDir.FullName, "models");

            if (Directory.Exists(modelsDir))
            {
                return currentDir.FullName;
            }

            if (currentDir.Parent == null)
            {
                break;
            }

            currentDir = currentDir.Parent;
        }

        throw new FileNotFoundException("Could not locate repository root with 'models' directory");
    }

}
