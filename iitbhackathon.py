import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import hashlib
import time
import random
import json

# Blockchain Class for transparency and accountability
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(previous_hash='1', proof=100)

    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'proof': proof,
            'previous_hash': previous_hash,
        }
        self.chain.append(block)
        return block

    def get_previous_block(self):
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = random.randint(1, 100000)
        while not self.is_valid_proof(previous_proof, new_proof):
            new_proof += 1
        return new_proof

    def is_valid_proof(self, previous_proof, new_proof):
        guess = f'{previous_proof}{new_proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:5] == '00000'

    def display_chain(self):
        return pd.DataFrame(self.chain)

# Simulated Dataset for Behavior Analysis
data = {
    'playtime': [2, 5, 1, 8, 3, 6, 4, 2, 7, 5],
    'spending': [10, 50, 5, 100, 20, 60, 40, 10, 80, 50],
    'age': [25, 30, 22, 35, 27, 31, 29, 24, 32, 28],
    'game_type': ['skill', 'chance', 'skill', 'chance', 'skill', 'chance', 'skill', 'chance', 'skill', 'chance'],
}
df = pd.DataFrame(data)

X = df[['playtime', 'spending', 'age']]
y = df['game_type'].apply(lambda x: 1 if x == 'chance' else 0)  # 1 for chance-based, 0 for skill-based

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML Model to Predict Game Type
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Responsible Gaming Reward System
def gamify_responsible_gaming(playtime, spending):
    thresholds = [
        (3, 50, 10),
        (5, 100, 5),
    ]
    for t_playtime, t_spending, points in thresholds:
        if playtime <= t_playtime and spending <= t_spending:
            print(f"Great job! You earned {points} reward points for responsible gaming.")
            return points
    print("Try reducing playtime or spending to earn more reward points.")
    return 0

# Real-time Alert for High Activity
def real_time_alert(playtime, spending):
    if playtime > 6 or spending > 100:
        return "Alert: High gaming activity detected! Consider taking a break."
    return "Gaming behavior is normal."

# Predict Game Type (Skill vs Chance)
def predict_game_type(user_data):
    prediction = model.predict([user_data])
    return 'Chance-based Game' if prediction == 1 else 'Skill-based Game'

# Transaction Logging (Blockchain Integration)
def log_transaction(transaction_details, blockchain):
    previous_block = blockchain.get_previous_block()
    proof = blockchain.proof_of_work(previous_block['proof'])
    blockchain.create_block(proof, previous_block['proof'])
    print(f"Transaction Logged: {transaction_details}")

# Visualization of Gaming Behavior
def visualize_behavior(users):
    """
    Visualize the behavior of users based on playtime, spending, and reward points.
    users: List of dictionaries containing user data.
    """
    # Create a DataFrame from the user data
    df = pd.DataFrame(users)

    # Ensure that reward points are included in the DataFrame
    if 'reward_points' not in df.columns:
        df['reward_points'] = 0  # Add a default column for reward points if missing

    # Plot behavior trends
    ax = df[['playtime', 'spending', 'reward_points']].plot(
        kind='bar',
        figsize=(10, 6),
        title="User Behavior Visualization",
        xlabel="User Index",
        ylabel="Values",
        stacked=True,
    )

    # Customize the x-axis to display user indices
    ax.set_xticks(range(len(users)))
    ax.set_xticklabels([f"User {i+1}" for i in range(len(users))], rotation=45)

    plt.legend(["Playtime (hrs)", "Spending ($)", "Reward Points"])
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()

# Simulate Users and Monitor Behavior
blockchain = Blockchain()
print("Welcome to the Enhanced Gaming Behavior Monitoring System!")

user_simulation = [
    {'playtime': 2, 'spending': 20, 'age': 26},
    {'playtime': 5, 'spending': 50, 'age': 30},
    {'playtime': 7, 'spending': 120, 'age': 22},
]

# Simulate user behavior and blockchain transactions
for user_data in user_simulation:
    print(f"\nAnalyzing user data: {user_data}")
    game_type = predict_game_type([user_data['playtime'], user_data['spending'], user_data['age']])
    print(f"Predicted Game Type: {game_type}")
    reward_points = gamify_responsible_gaming(user_data['playtime'], user_data['spending'])
    user_data['reward_points'] = reward_points  # Add reward points to the user data
    alert = real_time_alert(user_data['playtime'], user_data['spending'])
    print(f"Alert: {alert}")
    transaction = {'playtime': user_data['playtime'], 'spending': user_data['spending'], 'reward_points': reward_points}
    log_transaction(transaction, blockchain)

# Display Blockchain Data
print("\nBlockchain Data:")
print(blockchain.display_chain())

# Visualize User Behavior
visualize_behavior(user_simulation)
