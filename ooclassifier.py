# Copyright 2020 Paul Lu
import sys
import copy     # for deepcopy()
import string

Debug = False   # Sometimes, print for debugging
InputFilename = "file.input.txt"
TargetWords = [
        'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
        'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
        'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
        '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
        ]

label = "#help"
TargetWords = ["help", "get", "bad", "911"]

def open_file(filename=InputFilename):
    try:
        f = open(filename, "r")
        return(f)
    except FileNotFoundError:
        # FileNotFoundError is subclass of OSError
        if Debug:
            print("File Not Found")
        return(sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return(sys.stdin)


def safe_input(f=None, prompt=""):
    try:
        # Case:  Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # Case:  From file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF before strip()
                if Debug:
                    print("EOF")
                return("", False)
        return(line.strip(), True)
    except EOFError:
        return("", False)


class C274:
    def __init__(self):
        self.type = str(self.__class__)
        return

    def __str__(self):
        return(self.type)

    def __repr__(self):
        s = "<%d> %s" % (id(self), self.type)
        return(s)


class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        # FIXME:  Call superclass, here and for all classes
        self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        return

    def get_TF(self):
        return(self.TP, self.FP, self.TN, self.FN)

    # FIXME:  Use Python properties
    #     https://www.python-course.eu/python3_properties.php
    def set_target_words(self, lw):
        # Could also do self.targetWords = lw.copy().  Thanks, TA Jason Cannon
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return(self.targetWords)

    def get_allWords(self):
        return(self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return(self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return(self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords Hardcoded (%d): " % ln, end='')
        print(self.get_target_words())
        return

    def print_run_info(self):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag+"-------- Confusion Matrix --------")
        print(tag+"%10s | %13s" % ('Predict', 'Label'))
        print(tag+"-----------+----------------------")
        print(tag+"%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag+"%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag+"%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            print("TW %s: ( %10s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():    # FIXME Write predicate
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w            # FIXME Use first word, but change
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return(inClass, evidence)

    # Could use a decorator, but not now
    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return(cl, e)
    def classify_all(self, tset, update=False, tlabel="last"):
        for ti in tset.get_instances():
            cl, e = self.classify_by_words(ti, update, tlabel)

class ClassifyByTopN(ClassifyByTarget):
    def __init__(self, lw=[]):
        super().__init__(lw=lw)

    def target_top_n(self, tset, num=5, label=""):
        global TargetWords
        word_freq = {}
        instances = tset.get_instances()
        for i in instances:
            label_from_ti = i.inst["label"]
            if label == label_from_ti:
                for j in range(0, len(i.inst["words"])):
                    if i.inst["words"][j][0] == "#":
                        i.inst["words"][j] = "".join(i.inst["words"][j][1:])


                    if i.inst["words"][j] not in word_freq:
                        word_freq[i.inst["words"][j]] = 1
                    else:
                        word_freq[i.inst["words"][j]] += 1
        print(word_freq)
        TargetWords = []
        word_freq = sorted(word_freq.items(), key=lambda x: x[1])
        for i in range(1, len(word_freq)+1):
            if i <= num or word_freq[-i][1] == word_freq[-i+1][1]:
                TargetWords.append(word_freq[-i][0])
            else:
                break
        return




class TrainingInstance(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inst = dict()
        # FIXME:  Get rid of dict, and use attributes
        self.inst["label"] = "N/A"      # Class, given by oracle
        self.inst["words"] = []         # Bag of words
        self.inst["class"] = ""         # Class, by classifier
        self.inst["explain"] = ""       # Explanation for classification
        self.inst["experiments"] = dict()   # Previous classifier runs
        return

    def get_label(self):
        return(self.inst["label"])

    def get_words(self):
        return(self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):             # tlabel = tag label
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
                self, line, run=None,
                tlabel="read", inclLabel=True
            ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                # FIXME: For testing only.  Compare to previous version.
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return(self)
    
    def preprocess_words(self, mode=''):                                    
        if (mode != "keep-stops" and mode != "keep-digits" 
            and mode != "keep-symbols" and mode != ''):
            print("wtf")
        for w in range(1, len(self.inst["words"])):  
            self.inst["words"][w] = self.inst["words"][w].lower()  
            if mode != "keep-symbols":
                self.inst["words"][w] = self.rem_punc(self.inst["words"][w])    
            if mode != "keep-digits":
                self.inst["words"][w] = self.rem_num(self.inst["words"][w])
        if mode != "keep-stops":
            self.inst["words"] = self.rem_stop(self.inst["words"])
            
            
        return

    def rem_punc(self, word):                                               
        processed = []                                                      
        for letter in word:                                                 
            if letter not in string.punctuation:                            
                processed.append(letter)                                    
        return("".join(processed))                                          
    def rem_num(self, word):
        out = word                                                          
        if not word.isdigit():                                              
            out = []                                                        
            for letter in word:                                             
                if not letter.isdigit():                                       
                    out.append(letter)                                      
            out = "".join(out)                                              
        return out                                                          
    
    def rem_stop(self, ti_words):
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
"yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
"hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
"themselves", "what", "which","who", "whom", "this", "that", "these", "those",
"am", "is", "are", "was", "were", "be","been", "being", "have", "has", "had",
"having", "do", "does", "did", "doing", "a", "an","the", "and", "but", "if",
"or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
"about", "against", "between", "into", "through", "during", "before", "after",
"above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
"under", "again", "further", "then", "once", "here", "there", "when", "where",
"why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
"some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
"too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        
        out = []
        for i in ti_words:
            if i not in stopwords:
                out.append(i)
        return out


class TrainingSet(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inObjList = []     # Unparsed lines, from training set
        self.inObjHash = []     # Parsed lines, in dictionary/hash
        return

    def get_instances(self):
        return(self.inObjHash)      # FIXME Should protect this more

    def get_lines(self):
        return(self.inObjList)      # FIXME Should protect this more

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")     # Not used
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            # Check for comments
            if line[0] == '%':  # Comments must start with %
                continue

            # Save the training data input, by line
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    def preprocess(self, mode=''):
        for i in range(len(self.inObjHash)):
            self.inObjHash[i].preprocess_words(mode=mode)
            #self.update_lines(self.inObjHash[i].inst["words"], i)   #
                                                                    #        
    def update_lines(self, list_of_words, i):                       #
        self.inObjList[i] = " ".join(list_of_words)                 #

    def return_nfolds(self, num=3):
        lst_ts = []
        lst_inObjHash = []
        lst_inObjList = [] 
        for i in range(num):
            lst_inObjHash.append([])
            lst_inObjList.append([])
        index = 0
        for i in range(len(self.inObjHash)):
            if index == num:
                index = 0
            lst_inObjHash[index].append(self.inObjHash[i])
            lst_inObjList[index].append(self.inObjList[i])
            index += 1
        
        for i in range(num):
            ts = TrainingSet()
            ts.inObjHash = lst_inObjHash[i]
            ts.inObjList = lst_inObjList[i]
            lst_ts.append(ts)

        return lst_ts

def basemain():
    global label
    label = label

    tset = TrainingSet()
    run1 = ClassifyByTopN(TargetWords)
    print(run1)     # Just to show __str__
    lr = [run1]
    print(lr)       # Just to show __repr__

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or default filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    
    run1.classify_all(tset, update=True)        
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, label)


    tset.preprocess()
    run1.target_top_n(tset, num=3, label=label)
    if "".join(label[1:]) in TargetWords:
        TargetWords.remove("".join(label[1:]))
    run1.set_target_words(TargetWords)
    run1.classify_all(tset, update=True)
    run1.print_config()
    run1.eval_training_set(tset, label)


    tss = tset.return_nfolds(2)

    for i in range(2):
        
        #run1.target_top_n(tss[i], num=3, label="#weather")
        #run1.set_target_words(TargetWords)
        run1.classify_all(tss[i], update=True)
        print(TargetWords)

        if Debug:
            tset.print_training_set()
        run1.print_config()
        run1.print_run_info()
        run1.eval_training_set(tss[i], label)
    return


if __name__ == "__main__":
    basemain()
    
