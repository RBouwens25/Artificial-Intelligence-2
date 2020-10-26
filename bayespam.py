import argparse
import os
import re, math

from enum import Enum

class MessageType(Enum):
    REGULAR = 1,
    SPAM = 2

class Counter():

    def __init__(self):
        self.counter_regular = 0
        self.counter_spam = 0

    def increment_counter(self, message_type):
        """
        Increment a word's frequency count by one, depending on whether it occurred in a regular or spam message.

        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            self.counter_regular += 1
        else:
            self.counter_spam += 1

class Bayespam():

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.vocab = {}
        self.class_conditional_regular = {}
        self.class_conditional_spam = {}
        self.regular_correct = 0
        self.regular_wrong = 0
        self.spam_correct = 0
        self.spam_wrong = 0

    def list_dirs(self, path):
        """
        Creates a list of both the regular and spam messages in the given file path.

        :param path: File path of the directory containing either the training or test set
        :return: None
        """
        # Check if the directory containing the data exists
        if not os.path.exists(path):
            print("Error: directory %s does not exist." % path)
            exit()

        regular_path = os.path.join(path, 'regular')
        spam_path = os.path.join(path, 'spam')

        # Create a list of the absolute file paths for each regular message
        # Throws an error if no directory named 'regular' exists in the data folder
        try:
            self.regular_list = [os.path.join(regular_path, msg) for msg in os.listdir(regular_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'regular'." % path)
            exit()

        # Create a list of the absolute file paths for each spam message
        # Throws an error if no directory named 'spam' exists in the data folder
        try:
            self.spam_list = [os.path.join(spam_path, msg) for msg in os.listdir(spam_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'spam'." % path)
            exit()

    def read_messages(self, message_type):
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
        else:
            message_list = []
            print("Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')
                # Loop through each line in the message
                for line in f:
                    ## using lower() to force all letters into lowercase.
                    line = line.lower()
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line)):
                        token = split_line[idx]
                        ##Remove all non-alphanumeric characters from the string.
                        token = re.sub(r'[^a-z-]', '', token)

                        ##Only handles tokens with more than 3 characters
                        if(len(token) > 3):
                            if token in self.vocab.keys():
                                # If the token is already in the vocab, retrieve its counter
                                counter = self.vocab[token]
                            else:
                                # Else: initialize a new counter
                                counter = Counter()

                            # Increment the token's counter by one and store in the vocab
                            counter.increment_counter(message_type)
                            self.vocab[token] = counter

            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()

    def read_words(self, msg):
        ## This method reads a single message and returns a list of words in this message
        words = list()
        try:
            # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
            f = open(msg, 'r', encoding='latin1')
            # Loop through each line in the message
            for line in f:
                ## using lower() to force all letters into lowercase.
                line = line.lower()
                # Split the string on the space character, resulting in a list of tokens
                split_line = line.split(" ")
                # Loop through the tokens
                for idx in range(len(split_line)):
                    token = split_line[idx]
                    ##Remove all non-alphanumeric characters from the string.
                    token = re.sub(r'[^a-z-]', '', token)

                    ##Only handles tokens with more than 3 characters
                    if(len(token) > 3):
                        words.append(token)

        except Exception as e:
            print("Error while reading message %s: " % msg, e)
            exit()

        return words

    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages.

        :return: None
        """
        for word, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | In regular: %d | In spam: %d" % (repr(word), counter.counter_regular, counter.counter_spam))

    def write_vocab(self, destination_fp, sort_by_freq=False):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.

        :param destination_fp: Destination file path of the vocabulary file
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: None
        """

        if sort_by_freq:
            vocab = sorted(self.vocab.items(), key=lambda x: x[1].counter_regular + x[1].counter_spam, reverse=True)
            vocab = {x[0]: x[1] for x in vocab}
        else:
            vocab = self.vocab

        try:
            f = open(destination_fp, 'w', encoding="latin1")

            for word, counter in vocab.items():
                # repr(word) makes sure that special  characters such as \t (tab) and \n (newline) are printed.
                f.write("%s | In regular: %d | In spam: %d\n" % (repr(word), counter.counter_regular, counter.counter_spam),)

            f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)

    def train(self):
        ##Saved the count rather than calculate twice
        n_messages_regular = len(self.regular_list)
        n_messages_spam = len(self.spam_list)
        n_messages_total = n_messages_regular + n_messages_spam

        # print("N regular messages: ", n_messages_regular)
        # print("N spam messages: ", n_messages_spam)
        # print("N all messages: ", n_messages_total)

        ## compute a priori class possibilities
        priori_regular = n_messages_regular / n_messages_total
        priori_regular = math.log(10, priori_regular)
        priori_spam = n_messages_spam / n_messages_total
        priori_spam = math.log(10, priori_spam)

        # print("a priori regular: ", priori_regular)
        # print("a priori spam: ", priori_spam)

        n_words_regular = 0
        n_words_spam = 0

        for word, counter in self.vocab.items():
            n_words_regular += counter.counter_regular
            n_words_spam += counter.counter_spam

        tuner = 1 / (n_words_spam + n_words_regular)
        tuner = math.log(10,tuner)

        for word, counter in self.vocab.items():
            temp_reg = counter.counter_regular / n_words_regular
            temp_spam = counter.counter_spam/n_words_spam
            if temp_reg != 0:
                temp_reg = math.log(10, temp_reg)
                self.class_conditional_regular[word] = temp_reg
            else:
                self.class_conditional_regular[word] = tuner
            if temp_spam != 0:
                temp_spam = math.log(10, temp_spam)
                self.class_conditional_spam[word] = temp_spam
            else:
                self.class_conditional_spam[word] = tuner

    def test(self):
        ## The constant alpha is the same for regular and spam messages, and we want to compare regular and spam messages,
        ## so we will not compute alpha, because this is not needed.

        ## Test every message in regular list
        for msg in self.regular_list:
            ## The probabilities for this message start with the probability of any message to me regular or spam, which is 50/50
            msg_prob_regular = math.log(10, 0.5)
            msg_prob_spam = math.log(10, 0.5)

            ## Get all the words in this message
            message = self.read_words(msg)

            ## To compute the probabilities for this message, we need to add the probabilities for every word together
            for word in message:
                if word in self.class_conditional_regular:
                    msg_prob_regular += self.class_conditional_regular[word]
                if word in self.class_conditional_spam:
                    msg_prob_spam += self.class_conditional_spam[word]

            ## Classify the mail using the probabilities
            if msg_prob_regular > msg_prob_spam:
                self.regular_correct += 1
            else:
                self.regular_wrong += 1

        ## Test every message in spam list
        for msg in self.spam_list:
            ## The probabilities for this message start with the probability of any message to me regular or spam, which is 50/50
            msg_prob_regular = math.log(10, 0.5)
            msg_prob_spam = math.log(10, 0.5)

            ## Get all the words in this message
            message = self.read_words(msg)

            ## To compute the probabilities for this message, we need to add the probabilities for every word together
            for word in message:
                if word in self.class_conditional_regular:
                    msg_prob_regular += self.class_conditional_regular[word]
                if word in self.class_conditional_spam:
                    msg_prob_spam += self.class_conditional_spam[word]

            ## Classify the mail using the probabilities
            if msg_prob_regular > msg_prob_spam:
                self.spam_wrong += 1
            else:
                self.spam_correct += 1

        all_msg = len(self.regular_list) + len(self.spam_list)
        self.regular_correct = self.regular_correct/all_msg
        self.regular_wrong = self.regular_wrong/all_msg
        self.spam_correct = self.spam_correct/all_msg
        self.spam_wrong = self.spam_wrong/all_msg

        accuracy = self.regular_correct + self.spam_correct
        print("True regular: ", self.regular_correct)
        print("False spam: ", self.regular_wrong)
        print("True spam: ", self.spam_correct)
        print("False regular: ", self.spam_wrong)
        print("Accuracy: ", accuracy)


def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Read the file path of the folder containing the training set from the input arguments
    train_path = args.train_path

    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    bayespam.read_messages(MessageType.REGULAR)
    # Parse the messages in the spam message directory
    bayespam.read_messages(MessageType.SPAM)

    # bayespam.print_vocab()
    bayespam.write_vocab("vocab.txt")

    ## Train
    bayespam.train()

    ## Read the file path of the folder containing the testing set from the input arguments
    test_path = args.test_path

    ## Change the list of the regular and spam message locations to the test folder
    bayespam.list_dirs(test_path)

    # bayespam.print_vocab()
    bayespam.write_vocab("vocab.txt")

    ## Test
    bayespam.test()


    """
    Now, implement the follow code yourselves:
    check 1) A priori class probabilities must be computed from the number of regular and spam messages
    check 2) The vocabulary must be clean: punctuation and digits must be removed, case insensitive
    check 3) Conditional probabilities must be computed for every word
    check 4) Zero probabilities must be replaced by a small estimated value
    5) Bayes rule must be applied on new messages, followed by argmax classification
    6) Errors must be computed on the test set (FAR = false accept rate (misses), FRR = false reject rate (false alarms))
    7) Improve the code and the performance (speed, accuracy)

    Use the same steps to create a class BigramBayespam which implements a classifier using a vocabulary consisting of bigrams
    """

if __name__ == "__main__":
    main()
