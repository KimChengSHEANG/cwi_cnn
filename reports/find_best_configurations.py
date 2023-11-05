import glob
import os
import operator

def extract_accuracy(filename):
    with open(filename) as freport:
        content = freport.read()
        content = content.split("Accuracy: ")[1]
        accuracy = content.split("f1")[0].strip()

        id = filename.split('/')[1].split('_')[0]
        # print(accuracy)
        # print(id)
    return id, accuracy
def extract_reports(path):
    accuracies = {}
    for f in glob.glob(path):
        # print(f)
        id, acc = extract_accuracy(f)
        # print(id, acc)
        accuracies[id] = acc
        # break
    return accuracies

def extract_best_score(top):
    News_Test = extract_reports("News_Test/*.txt")
    WikiNews_Test = extract_reports("WikiNews_Test/*.txt")
    Wikipedia_Test = extract_reports("Wikipedia_Test/*.txt")

    All_Test = {}
    All_Sum = {}
    for key in News_Test:
        item = [News_Test[key], WikiNews_Test[key], Wikipedia_Test[key]]
        All_Test[key] = item
        All_Sum[key] = float(News_Test[key]) + \
            float(WikiNews_Test[key]) + float(Wikipedia_Test[key])

    # print(len(All_Test))
    sorted_items = sorted(All_Sum.items(), key=operator.itemgetter(1))
    # sorted_items = sorted_items[-5:]
    # print(sorted_items)
    for (key, val) in sorted_items[-top:]:
        print('')
        print(News_Test[key], WikiNews_Test[key], Wikipedia_Test[key], "key =", key, "Sum =", val)
def show_configurations(key):
    with open("News_Test/" + key + "_report.txt") as f:
        content = f.read()
        print(content.split("=======")[0])
    with open("WikiNews_Test/" + key + "_report.txt") as f:
        content = f.read()
        print(content.split("=======")[0])
    with open("Wikipedia_Test/" + key + "_report.txt") as f:
        content = f.read()
        print(content)

print("==============================================================================")
extract_best_score(10)
# show_configurations("1550504858")
# show_configurations("1550490508")
