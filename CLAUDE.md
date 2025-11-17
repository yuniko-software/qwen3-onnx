# CLAUDE.md

## General Recommendations

1) Don't add comments inside functions/methods unless critical. Prefer documentation comments for public members.
2) Don't create README or documentation markdown files unless I directly ask you to.
3) In comments or READMEs (if requested), avoid words like "enhanced," "smart," or "comprehensive." Use precise and unambiguous words.
4) If you make a complex change (multiple file refactoring, architecture changes, etc.), write a plan before implementing to get confirmation.
5) When researching unknown libraries or code, go directly to GitHub repositories and official documentation instead of guide websites or tutorials. GitHub repositories provide accurate, up-to-date information from the source.

## When Working with .NET

1) Use C# 14 syntax.
2) Prefer immutable structures like records and read-only collections when it makes sense.
3) After each change, run `dotnet format`.
