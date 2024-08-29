import csv

# Define a list of example destinations and durations
destinations = [
    "Amsterdam", "Paris", "Tokyo", "New York", "London", "Italy", "Holland",
    "Berlin", "Madrid", "Brussels", "Budapest", "Munich", "Lisbon", "Russia",
    "Greece"
]

durations = ["1 week", "a weekend"]

# Generate 100 requests
requests = []
for i in range(100):
    destination = destinations[i % len(destinations)]
    duration = durations[i % len(durations)]
    user_request = f"Give me a plan for {duration} to {destination}"
    requests.append({"user_request": user_request, "destination": destination})

# Write the requests to a CSV file
csv_file = "datasets/test_requests_for_UseCase1/travel_requests.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["user_request", "destination"])
    writer.writeheader()
    writer.writerows(requests)

print(f"CSV file '{csv_file}' has been created with 100 travel requests.")
