# Conventional commits (reminder)
*type(scope): description*

A commit is a brief and descriptive summary of codebase changes for an informative git log. It also enables the use of automated tools for creating changelogs, semantic versioning etc. 

## Type

**feat**: new feature  
**fix**: bug fix  
**refactor**: rewrite or restructuring of code  
**perf**: performance improvements  
**build**: changes to tools, dependencies, docker, ci/cd etc  
**test**: add missing or correct existing tests  
**style**: code style changes (formatting, type hints etc)  
**docs**: documentation changes  
**misc**: miscellaneous changes (codebase structure, gitignore etc)  

## Scope (optional)
Add a 'one-word' scope for context within parenthesis. 

*Examples*  

feat **(database)** : add new logic to filters  
fix **(env-tokens)** : sync the use of tokens across scripts  

## Description
Briefly describe your commit. 

*Guidelines*

- the entire message should be in lowercase
- no final punctuation
- imperative - "add link" not "added" or "adds", think of it like "This commit will ..."
- soft limit of 50 characters


## Breaking changes
A commit that introduces breaking changes to the project. A breaking change can be part of any type and is denoted by an exclamation mark. 

*Examples*

build **!** : change dependency versions to match essential api  
fix(parser) **!** : introduce new parser logic 

**Note**: it's recommended to use single-quotes (' ') for breaking changes. Otherwise some shells like bash and zsh interpret the exclamation mark as a command to run previous commands from history. 