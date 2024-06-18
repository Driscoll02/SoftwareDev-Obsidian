pip is by far the most common installer for python packages. This is very similar to npm in JavaScript and Node.js.
## Command list:

```
pip list : lists all currently installed packages

pip install <package name> : installs a new package

pip uninstall <package name> : uninstalls an installed package

pip freeze > requirements.txt : saves the list of all installed packages and their                                 versions to a requirements.txt file

pip install -r requirements.txt : installs all packages listed in file. -r tells                                     the command to read from a file
```