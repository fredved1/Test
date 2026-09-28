# FC Twente ticket monitor

Kijkt continu op de ticketshop of er plekken vrijkomen in de vakken die je wilt,
en stuurt een Telegram-bericht (optioneel ook ntfy). **Koopt niet zelf**: je krijgt een alert en koopt zelf.

## Installatie op de VPS (Ubuntu/Debian)

```bash
sudo git clone -b claude/twente-ajax-ticket-scraper-m9vrsu https://github.com/fredved1/Test.git /opt/Test
sudo ln -s /opt/Test/twente_tickets /opt/twente_tickets
cd /opt/twente_tickets
sudo python3 -m venv venv
sudo venv/bin/pip install -r requirements.txt
sudo venv/bin/playwright install --with-deps chromium
sudo cp .env.example .env && sudo nano .env      # vul inlog, vakken en Telegram in (zie hieronder)
sudo chmod 600 .env
```

## Telegram-bot aanmaken (5 minuten)

1. Open Telegram, zoek **@BotFather** (blauw vinkje) en stuur `/newbot`.
2. Geef een naam (bijv. `Twente Tickets`) en een gebruikersnaam die op `bot` eindigt
   (bijv. `twente_tickets_thomas_bot`).
3. BotFather stuurt je een **token** zoals `123456789:AAH...`. Zet die in `.env` als
   `TELEGRAM_BOT_TOKEN=...`. Deel dit token met niemand: wie het heeft, kan je bot besturen.
4. Klik in het bericht van BotFather op de link naar je bot en stuur hem `/start`.
   (Zonder deze stap mag de bot jou geen berichten sturen.)
5. Haal je chat-ID op:
   ```bash
   sudo venv/bin/python monitor.py telegram-chat-id
   ```
   Zet de regel die hij print (`TELEGRAM_CHAT_ID=...`) in `.env`.
6. Test:
   ```bash
   sudo venv/bin/python monitor.py test-alert
   ```
   Je hoort nu een bericht "Test" van je bot te krijgen. Zet in Telegram de meldingen
   voor deze chat aan (niet dempen), zodat je het direct hoort.

## Stap 1: discover (verplicht)

De pagina-opbouw van de shop is vooraf niet bekend. Draai eerst:

```bash
sudo venv/bin/python monitor.py discover
```

Daarna staan in `state/discover/` de `screenshot.png`, `page.txt`, `page.html` en `responses.json`.
Kijk hoe de vakken heten (bijv. "West", "Oost", "Vak 104") en zet die in `SECTION_KEYWORDS`.
Staat er "0 seat-like objects" in de output, dan laadt de stoelenkaart pas na een klik op een vak;
in dat geval moet het script nog aangepast worden aan die pagina (stuur de discover-bestanden door).

## Stap 2: continu draaien

```bash
sudo cp twente-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now twente-monitor
journalctl -u twente-monitor -f     # logs bekijken
```

## Soorten alerts

- **"2 plekken naast elkaar vrij!"**: uit de stoelendata zijn aansluitende vrije stoelen in een gewenst vak gevonden.
- **"Mogelijk kaarten beschikbaar"**: een tekstregel noemt een gewenst vak en zegt niet "uitverkocht".
- **"Ticketpagina veranderd"**: de pagina is veranderd, maar zonder duidelijke match. Kijk dan zelf even.
- **"Twente monitor faalt"**: 5 checks achter elkaar mislukt (inlog/blokkade/site down).
