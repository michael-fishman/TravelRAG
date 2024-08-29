import csv

# Define a list of example destinations and durations
destinations = [
    "Amsterdam", "Paris", "Tokyo", "New York", "London", "Rome", "Sydney",
    "Berlin", "Barcelona", "Dubai", "Moscow", "Los Angeles", "Toronto",
    "Vienna", "Bangkok", "Singapore", "Istanbul", "Prague", "Seoul", "Madrid"
]

durations = [
    "2 weeks", "1 week", "10 days", "5 days", "3 days", "a weekend", "4 days"
]

# Generate 100 requests
requests = []
for i in range(100):
    destination = destinations[i % len(destinations)]
    duration = durations[i % len(durations)]
    user_request = f"Give me a plan for {duration} to {destination}"
    requests.append({"user_request": user_request, "destination": destination})

# Write the requests to a CSV file
csv_file = "travel_requests.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["user_request", "destination"])
    writer.writeheader()
    writer.writerows(requests)

print(f"CSV file '{csv_file}' has been created with 100 travel requests.")
