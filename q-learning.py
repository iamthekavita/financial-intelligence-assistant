# import random

# #Actions
# actions = ["left", "right"]

# #Rewards
# rewards= {
#     "left": 0,
#     "right": 1
# }

# # AI tries multiple times
# for i in range(5):
#     #Random Action
#     action = random.choice(actions)    #AI randomly select action

#     #Get reward
#     reward = rewards[action]

#     print("Action:", action)
#     print("Reward:", reward)

#     if reward == 1:
#         print("Good action")
#     else:
#         print("Bad action")






import streamlit as st

st.title("My Streamlit App")

name = st.text_input("Enter your name")

if name:
    st.write("Hello", name)