import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PORTFOLIO] - %(message)s')

class PortfolioManager:
    def __init__(self, api_client):
        """
        Manages Balance Checks and Position Sizing.
        """
        self.client = api_client
        self.usd_account_id = None
        self.btc_account_id = None
        
        # We cache the Account IDs so we don't have to search for them every single time
        self._find_account_ids()

    def _find_account_ids(self):
        """
        Fetches all accounts and finds the specific IDs for USD and BTC.
        """
        try:
            response = self.client.get_accounts()
            accounts = response.accounts
            
            for acc in accounts:
                if acc.currency == "USD":
                    self.usd_account_id = acc.uuid
                elif acc.currency == "BTC":
                    self.btc_account_id = acc.uuid
                    
            logging.info(f"✅ Accounts Linked. USD: ...{str(self.usd_account_id)[-4:]} | BTC: ...{str(self.btc_account_id)[-4:]}")
            
        except Exception as e:
            logging.error(f"Failed to fetch account IDs: {e}")

    def get_balances(self):
        """
        Returns the CURRENT Available Balance for USD and BTC.
        """
        try:
            response = self.client.get_accounts()
            
            usd_bal = 0.0
            btc_bal = 0.0
            
            for acc in response.accounts:
                if acc.currency == "USD":
                    usd_bal = float(acc.available_balance.value)
                elif acc.currency == "BTC":
                    btc_bal = float(acc.available_balance.value)
                    
            return usd_bal, btc_bal
            
        except Exception as e:
            logging.error(f"Failed to get balances: {e}")
            return 0.0, 0.0

    def get_buy_size(self, safety_buffer=0.99):
        """
        Calculates how much USD we can spend, leaving room for fees.
        Default buffer is 1% (0.99 multiplier).
        """
        usd_bal, _ = self.get_balances()
        
        # Logic: If we have $1000, we only trade $990 to pay for the 0.6% taker or 0.2% maker fee safely.
        safe_amount = usd_bal * safety_buffer
        
        # Coinbase minimum is usually around $1-2, so let's enforce a floor.
        if safe_amount < 5.00:
            logging.warning(f"Insufficient funds to buy (${usd_bal}). Minimum $5 required.")
            return 0.0
            
        return round(safe_amount, 2)

    def get_sell_size(self):
        """
        Returns 100% of the BTC balance available for selling.
        """
        _, btc_bal = self.get_balances()
        
        # Check for dust (tiny amounts less than 0.00001 BTC)
        if btc_bal < 0.00001:
            logging.warning(f"Insufficient BTC to sell ({btc_bal}).")
            return 0.0
            
        # We truncate to 8 decimals to be safe with API limits
        return float(f"{btc_bal:.8f}")