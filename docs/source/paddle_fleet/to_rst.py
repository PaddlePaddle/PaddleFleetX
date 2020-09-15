import os

files = os.listdir(".")
print(files)
for f in files:
    if ".md" in f:
        os.system("pandoc {} -f markdown -t rst -o ../paddle_fleet_rst/{}".format(f, f.replace(".md", ".rst")))
