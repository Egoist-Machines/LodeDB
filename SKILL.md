# Memory Skill

This is a skill which is used for any function related to memory, it uses LodeDB 
for fast searching and the data completely stays on your local machine for privacy.

## Tools

- **save** — To save a memory on local memory.
- **recall** — To recall data with a given query from local memory.
- **forget** — To remove a memory from local memory.
- **stats** — To check the statistics of local memory.

## Memory Rules

- Always recall from local memory before answering a question.
- Only save memories that are important such as user preferences, decisions, or facts — not random chat.
- When a new preference is given, first forget the old one then save the new one.
- Only save data when it is not found when recalled from local memory.