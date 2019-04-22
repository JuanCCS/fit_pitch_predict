import sys
import json

def main(argv):
    f = open("myDemoFile.txt", "a")
    f.write("Now the file has one more line!")
    while True:
        line = sys.stdin.readline()
        if line is not None:
            if len(line) > 0:
                get_obj = json.loads(line)
                print(get_obj)
                f.close()
        return

if __name__ == "__main__":
   main(sys.argv)