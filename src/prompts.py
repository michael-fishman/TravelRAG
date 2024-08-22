import re

def get_travel_plan_prompt(user_request):
    # Basic pattern matching to identify key components in the user's request
    # TODO: change this part in the prompt bscause it doesnt recognize for example "weekend"
    # TODO: maybe just attach the full request without rephrsing it
    destination_pattern = r'\bto\s([\w\s]+)'
    duration_pattern = r'\bfor\s(\d+\s\w+)'

    # Search for destination
    destination_match = re.search(destination_pattern, user_request)
    destination = destination_match.group(1).strip() if destination_match else "the destination"

    # Search for duration
    duration_match = re.search(duration_pattern, user_request)
    duration = duration_match.group(1).strip() if duration_match else "a suitable duration"

    # Build the prompt based on identified components
    prompt = (
        f"Please create a detailed travel plan for {duration} to {destination}. "
        "The output should be in the following format:\n\n"
        "Travel Plan:\n"
        "[Here you provide a detailed day-by-day plan, including major landmarks. "
        "For each landmark, use a tag in the format <Landmark> where 'Landmark' is the name of the place. "
        "This tag will be used to insert photos later.]\n\n"
        "Example:\n"
        "<A>\n"
        "A: Visit the A landmark in the morning.\n"
        "<B>\n"
        "B: In the afternoon, go to B for sightseeing.\n\n"
        "Landmarks:\n"
        "[List all the landmarks mentioned in the plan, separated by commas.]\n\n"
        "Ensure that the 'Landmarks' section is clearly separated at the end of the plan."
    )
    return prompt

def get_location_recognizer_prompt(img_query):
    prompt = (
        "You are an advanced AI model specialized in recognizing and identifying landmarks and famous places from images. "
        "Analyze the image provided and return **only** the name of the most likely landmark or location depicted in the image. "
        "Do not provide any additional information, descriptions, or explanations."
        "The final answe should be retuned in the following dict format {'name': landmark_name, 'location': landmark_location}."
        "Here is the image:\n"
    )
    return [prompt, img_query]
    

if __name__ == "__main__":
    # Example usage - get_travel_plan_prompt
    user_request_1 = "Give me a plan for a trip for 2 weeks to Amsterdam"
    user_request_2 = "I need a weekend getaway plan to Paris"
    user_request_3 = "Suggest a vacation itinerary for 10 days to Tokyo"

    prompt_1 = get_travel_plan_prompt(user_request_1)
    prompt_2 = get_travel_plan_prompt(user_request_2)
    prompt_3 = get_travel_plan_prompt(user_request_3)

    print("\n-----\nGenerated Prompt 1:\n", prompt_1)
    print("\n-----\nGenerated Prompt 2:\n", prompt_2)
    print("\n-----\nGenerated Prompt 3:\n", prompt_3)

    # Example usage - get_location_recognizer_prompt
    # TODO: add example usage