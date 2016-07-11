'''
The agent base class as well as a baseline agent.
'''

from abc import abstractmethod
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Agent(object):
    def __init__(self, name):
        self.name = name
    
    def __repr__(self):
        return "Agent_" + self.name
    
    def will_buy(self, value, price, prob):
        """Given a value, price, and prob of Excellence,
        return True if you want to buy it; False otherwise.
        The rational agent.
        Do NOT change or override this."""
        return value*prob > price

    def train(self, X_train, y_train, X_val, y_val):
        """First, choose the best classifier that when trained
        on <X_train, y_train> performs the best on <X_val, y_val>.
        Then, train that classifier on X_train, y_train.
        Do NOT change or override this method.
        """
        self.clf = self.choose_the_best_classifier(X_train, y_train, X_val, y_val)

        # Train the classifier
        self.clf.fit(X_train, y_train)

        # Find the index of the Excellent class.
        # Will be used in predict_prob_of_excellent.
        if self.clf.classes_[0] == 'Excellent':
            self._excellent_index = 0
        else:
            self._excellent_index = 1

    @abstractmethod
    def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
        """This method choses the 'best' sklearn classifier, which
        when trained on <X_train, y_train> and performs
        the 'best' on <X_val, y_val>.
        Search over three classifiers:
        1. A BernoulliNB with default constructor.
        2. A LogisticRegression with default constructor.
        3. An SVC that has a degree 4 polynomial kernel and
        probability estimates are turned on.
        Your agent should choose the best classifier for the job.
        You should define what it means to be the 'best'.
        This method should return an untrained newly constructed
        object that is an instance of the best chosen classifier.
        OVERRIDE this method."""
        

    def predict_prob_of_excellent(self, x):
        """Given a single product, predict and return
        the probability of the product being Excellent.
        Do NOT change or override this method.       
        """
        return self.clf.predict_proba(x)[0][self._excellent_index]

class Agent_single_sklearn(Agent):
    """A baseline agent that simply uses a single classifier
    and does not search for the best classifier."""

    def __init__(self, name, clf):
        super(Agent_single_sklearn, self).__init__(name)
        self.clf = clf

    def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
        "Simply return the classifier that was provided to the constructor."
        return self.clf

class Agent_bsharma(Agent):
    def __init__(self,name):
        super(Agent_bsharma,self).__init__(name)
        self.fixed_prob = 1
    def train(self, X_train, y_train, X_val, y_val):
        self.clf = self.choose_the_best_classifier(X_train, y_train, X_val, y_val)

        # Train the classifier
        self.clf.fit(X_train, y_train)

        # Find the index of the Excellent class.
        # Will be used in predict_prob_of_excellent.
        if self.clf.classes_[0] == 'Excellent':
            self._excellent_index = 0
        else:
            self._excellent_index = 1

    def choose_the_best_classifier(self, X_train, y_train, X_val, y_val):
        """This method choses the 'best' sklearn classifier, which
        when trained on <X_train, y_train> and performs
        the 'best' on <X_val, y_val>.
        Search over three classifiers:
        1. A BernoulliNB with default constructor.
        2. A LogisticRegression with default constructor.
            3. An SVC that has a degree 4 polynomial kernel and
        probability estimates are turned on.
        Your agent should choose the best classifier for the job.
        You should define what it means to be the 'best'.
        This method should return an untrained newly constructed
        object that is an instance of the best chosen classifier.
        OVERRIDE this method."""

        row_X = X_val.shape[0] # row_y contains the number of sample provided in the train
        col_X = X_val.shape[1] # number of features available in train data
        val_vect = np.empty(shape = col_X) # vector of features to be used while predicting the probability
        output = np.empty(shape=2)
        predicted_result = np.empty(shape = row_X)
        clf= BernoulliNB() # object of BernoulliNB() class
        clf1 = LogisticRegression() # object of LogisticRegression() class
        clf2 = SVC(probability=True,kernel='poly',degree=4,random_state=0) # object of SVC() class
        clf.fit(X_train,y_train) #calling fit method to train each classifier
        clf1.fit(X_train,y_train)#calling fit method to train each classifier
        clf2.fit(X_train,y_train)#calling fit method to train each classifier

	"""we creates a local agent which follows a classifier.
	   We consider wealth as performance parameter and calculate the wealth gained by each agent on a data set.
	   we then compare wealth of classifiers
	   The classifier having the maximum wealth is the BEST ONE for that particular data set and its object will be passed to train 	function."""

        agent_wealths= {} # agent_wealths is an array for holding wealth for an agent for each of the classifier
        num_products = X_val.shape[0] # number of products available in the train data.
        value = 1000 #intial default value 
        price_trials = 10
        for agent in range(3):# Iterating for each classifier to find the wealth collected by each
            agent_wealths[agent] = 0 # assing 0 wealth to agent for each classifier 
            for outer_counter in range(row_X): # counter to iterate over each product
                excellent = (y_val[outer_counter] == 'Excellent') #boolean value for the quality of product being excellent
                for count in range(col_X): #making a vector for each set of features
                    val_vect[count] = X_val[outer_counter][count]
                    if agent == 0:
                        output = clf.predict_proba(val_vect)
                    elif agent == 1:
                        output = clf1.predict_proba(val_vect)
                    else:
                        output = clf2.predict_proba(val_vect)
                    prob = output[0][0] #output holds the probability of each product being excellent
                for pt in range(price_trials): #loop to calculate the wealth of each agent depending if it buys a product or not
                    price = ((2*pt+1)*value)/(2*price_trials)
                    if value*prob > price: # condition to buy a product
                        agent_wealths[agent] -= price
                        if excellent:
                            agent_wealths[agent] += value
#Below if-else function to find the which classifier/agent collects the maximum wealth.The classifier will be used for the particular data set.
        if (agent_wealths[0] > agent_wealths[1]) and (agent_wealths[0] > agent_wealths[2]):
	    clf_new= BernoulliNB() # creating new instance to send unlearned class
	    return clf_new
        elif (agent_wealths[1] > agent_wealths[0]) and (agent_wealths[1] > agent_wealths[2]):
            clf1_new = LogisticRegression() # creating new instance to send unlearned class
            return clf1_new
        else:
	    clf2_new = SVC(probability=True,kernel='poly',degree=4,random_state=0) # creating new instance to send unlearned class
            return clf2_new 

    def will_buy(self, value, price, prob):
        """Given a value, price, and prob of Excellence,
        return True if you want to buy it; False otherwise.
        The rational agent.
        Do NOT change or override this."""
        return value*prob > price
