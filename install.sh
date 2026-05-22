#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

printf "${GREEN}==================================================${NC}\n"
printf "${GREEN}Starting automated installation for cudallm-cli${NC}\n"
printf "${GREEN}==================================================${NC}\n"

if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        printf "${CYAN}[INFO] Activating existing virtual environment...${NC}\n"
        source .venv/bin/activate
    else
        printf "${CYAN}[INFO] Creating new virtual environment (.venv)...${NC}\n"
        python3 -m venv .venv
        source .venv/bin/activate
    fi
fi

printf "${YELLOW}[STEP 1/3] Upgrading pip...${NC}\n"
python3 -m pip install --upgrade pip

printf "${YELLOW}[STEP 2/3] Installing dependencies from requirements.txt...${NC}\n"
pip install -r requirements.txt --no-cache-dir

printf "${YELLOW}[STEP 3/3] Installing cudallm-cli in editable mode...${NC}\n"
pip install -e .

printf "${CYAN}[INFO] Initializing environment config...${NC}\n"
cudallm init

printf "${GREEN}==================================================${NC}\n"
printf "${GREEN}Installation completed! Run 'cudallm doctor' to verify.${NC}\n"
printf "${YELLOW}Please ensure you run 'source .venv/bin/activate' in your shell.${NC}\n"
printf "${GREEN}==================================================${NC}\n"
