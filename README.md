# UO_Physics_Machine_Learning

A basic tutorial on git can be found [here](https://www.atlassian.com/git/tutorials/). A quick overview of some useful git commands can be found [here](http://rogerdudler.github.io/git-guide/).

Some GUIs that can be used with git: [GitHub Desktop](https://desktop.github.com/), [SourceTree](https://www.sourcetreeapp.com/).

When developing code in this repository, personal branches should generally be based on the master branch, and then merged back into the master branch. For learning purposes, branch merging for this repository should not use fast forward merging in general in order to track work done from each person on personal branches. When merging branches, the '--no-ff' flag can be used to force a merge commit instead of a fast forward commit. Merge committing without fast forwarding can be set to repository wide default with the following: 'git config --add merge.ff false'. A fast forward commit can be allowed using the '--ff' flag. The '--ff-only' flag will only merge if a fast forward merge is possible.

Note: if a merge commit is set as the default, 'git pull' will create a new commit when run because it is the equivalent of running 'git fetch' followed by 'git merge'. Use 'git pull --ff-only' to allow fast forward merges when using 'git pull' or run 'git fetch' and 'git merge' individually with the desired options.
