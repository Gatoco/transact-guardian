#!/usr/bin/env python3
"""
Generate a secure API key and save to .env file.
"""
import secrets
import string
import os

def generate_api_key(length=32):
    """Generate a secure API key."""
    prefix = "fk_"
    chars = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(chars) for _ in range(length))
    return prefix + random_part

def main():
    api_key = generate_api_key()

    # Read existing .env or create new
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

    with open(env_path, 'r') as f:
        lines = f.readlines()

    # Update or add API_KEY line
    new_lines = []
    key_found = False
    for line in lines:
        if line.startswith('API_KEY='):
            new_lines.append(f'API_KEY={api_key}\n')
            key_found = True
        else:
            new_lines.append(line)

    if not key_found:
        new_lines.append(f'API_KEY={api_key}\n')

    with open(env_path, 'w') as f:
        f.writelines(new_lines)

    print(f"✓ API Key generated and saved to .env")
    print(f"  API_KEY={api_key}")
    print(f"\nCopy this key to use in your requests:")
    print(f'  curl -H "Authorization: Bearer {api_key}" ...')

if __name__ == "__main__":
    main()