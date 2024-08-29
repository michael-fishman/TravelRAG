def get_travel_plan_prompt(user_request):
    prompt = (
        "You are an advanced travel planning AI. Your task is to generate a detailed travel itinerary based on the user's request. "
        "Here are some key points to consider when creating the itinerary:\n\n"
        "- If the user specifies the duration of the trip (e.g., 2 weeks, 10 days), plan the itinerary for that time frame.\n"
        "- If the user does not mention a duration, plan a trip for a reasonable duration, such as 5-7 days.\n"
        "- If the user requests a plan for a specific landmark, create an itinerary that includes that landmark and suggests additional landmarks in the same country.\n"
        "- The itinerary should include key landmarks in the country that are relevant to the user's request.\n"
        "- The travel plan should be returned in a valid JSON format, structured as an object containing a list of days. Each day should be an object with the following fields:\n"
        "  * 'day_of_trip': The day number of the trip (as an integer).\n"
        "  * 'landmark': The name of the landmark and its location (e.g., 'Eiffel Tower, Paris, France').\n"
        "  * 'plan_description': A brief description of the plan for that day (e.g., 'Visit the Eiffel Tower and enjoy a picnic at the Champ de Mars').\n"
        "- Ensure the output is properly formatted as valid JSON, with no extraneous characters, code blocks, or newline escape sequences.\n"
        "- Do not include any surrounding backticks or labels like 'json'. The output should be a clean, parsable JSON object.\n\n"
        "Example output:\n"
        "{\n"
        "  \"itinerary\": [\n"
        "    {\"day_of_trip\": 1, \"landmark\": \"Eiffel Tower, Paris, France\", \"plan_description\": \"Visit the Eiffel Tower and enjoy a picnic at the Champ de Mars\"},\n"
        "    {\"day_of_trip\": 2, \"landmark\": \"Louvre Museum, Paris, France\", \"plan_description\": \"Explore the Louvre Museum and see the Mona Lisa and other famous artworks\"},\n"
        "    {\"day_of_trip\": 3, \"landmark\": \"Montmartre, Paris, France\", \"plan_description\": \"Wander around Montmartre, visit the Sacré-Cœur Basilica, and enjoy the artistic vibe of the area\"},\n"
        "    {\"day_of_trip\": 4, \"landmark\": \"Palace of Versailles, Versailles, France\", \"plan_description\": \"Take a day trip to the Palace of Versailles and explore the opulent gardens and royal apartments\"},\n"
        "    {\"day_of_trip\": 5, \"landmark\": \"Seine River, Paris, France\", \"plan_description\": \"Take a scenic cruise along the Seine River and enjoy views of iconic Parisian landmarks\"}\n"
        "  ]\n"
        "}\n\n"
        "Based on these guidelines, generate a travel itinerary for the following user request:\n"
        f"'{user_request}'"
    )

    return prompt


def get_location_recognizer_prompt(img_query):
    prompt = (
        "You are an advanced AI model specialized in recognizing and identifying landmarks and famous places from images. "
        "Analyze the image provided and return **only** the name of the most likely landmark or location depicted in the image. "
        "Do not provide any additional information, descriptions, or explanations."
        "The final answer should be returned in the following dict format {'name': landmark_name, 'location': landmark_location}."
        "Here is the image:\n"
    )
    return [prompt, img_query]


def get_prompt_for_creating_full_answer(user_name, landmark):
    landmark_answer_example = (
        "part 1: Hi <name>!\nIt looks like your picture is from <landmark>.\n"
        "part 2: You should definitely plan a trip there—it's a fantastic destination!\n"
        "part 3: You're also welcome to try our travel planner, TravelRAG, "
        "where you can explore the best images of more landmarks in that country!"
    )

    prompt = (
        "You are a creative and knowledgeable AI model. Your task is to generate a personalized message "
        "for a user based on the landmark identified from a photo they uploaded."
        "The user did not take the photo themselves; they used our system to identify where it was taken.\n"
        "Here is an example response:\n\n"
        f"{landmark_answer_example}\n\n"
        "Please generate a similar message, but vary the wording so that the response is unique each time. "
        "Make sure to include the following in your response:\n"
        "- If a name is provided, include a personalized greeting using the user's name.\n"
        "- Acknowledgment that the picture is from the identified landmark.\n"
        "- A suggestion for the user to visit the landmark, highlighting it as a great destination.\n"
        "- Mention the TravelRAG system in part 3, suggesting the user explore it for more landmarks in the same country.\n"
        "- Optionally, you can add an interesting fact, history, or travel tip related to the landmark to make the response richer.\n"
        "- If the name is not provided, craft the response without a personalized greeting.\n\n"
        "Inputs Description:\n"
        "- name: the name of the user (optional, might be None).\n"
        "- landmark: the name of the landmark identified from the image.\n"
        "Inputs:\n"
        f"- name: {user_name}\n"
        f"- landmark: {landmark}\n\n"
        "Generate the response now."
    )
    return prompt
