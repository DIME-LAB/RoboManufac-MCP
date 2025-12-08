from mcp.server.fastmcp import FastMCP
import json
from pathlib import Path

mcp = FastMCP("Inventory Management")

# Path to persistent inventory data file
INVENTORY_FILE = Path(__file__).parent / "data" / "inventory.json"

def load_inventory():
    """Load inventory from JSON file"""
    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventory file not found at {INVENTORY_FILE}. "
            "Please ensure data/inventory.json exists."
        )
    
    with open(INVENTORY_FILE, 'r') as f:
        return json.load(f)

def save_inventory(data):
    """Save inventory to JSON file"""
    # Ensure data directory exists
    INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ========== TOOLS FOR MANAGING INVENTORY ==========

@mcp.tool()
def get_inventory() -> str:
    """
    Get all inventory items with their details
    
    Returns:
        JSON string containing all inventory items
    """
    data = load_inventory()
    
    if not data.get("items"):
        return json.dumps({"items": [], "message": "Inventory is empty"}, indent=2)
    
    return json.dumps(data, indent=2)

@mcp.tool()
def get_item_by_id(inventory_id: str) -> str:
    """
    Get a specific inventory item by its ID
    
    Args:
        inventory_id: The ID of the inventory item to retrieve
    
    Returns:
        JSON string with item details or error message
    """
    data = load_inventory()
    
    for item in data.get("items", []):
        if item["id"] == inventory_id:
            return json.dumps(item, indent=2)
    
    available_ids = [item["id"] for item in data.get("items", [])]
    return json.dumps({
        "error": f"Item with ID '{inventory_id}' not found",
        "available_ids": available_ids
    }, indent=2)

@mcp.tool()
def get_item_by_name(name: str) -> str:
    """
    Get a specific inventory item by its name
    
    Args:
        name: The name of the inventory item to retrieve
    
    Returns:
        JSON string with item details or error message
    """
    data = load_inventory()
    
    for item in data.get("items", []):
        if item["name"].lower() == name.lower():
            return json.dumps(item, indent=2)
    
    available_names = [item["name"] for item in data.get("items", [])]
    return json.dumps({
        "error": f"Item with name '{name}' not found",
        "available_names": available_names
    }, indent=2)

@mcp.tool()
def update_inventory_price(inventory_id: str, new_price: str) -> str:
    """
    Update the price of an inventory item by its ID
    
    Args:
        inventory_id: The ID of the inventory item to update
        new_price: The new price (as a string, e.g., "9.99")
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_inventory()
    
    for item in data.get("items", []):
        if item["id"] == inventory_id:
            old_price = item["price"]
            item["price"] = new_price
            save_inventory(data)
            return json.dumps({
                "success": True,
                "message": f"Updated {item['name']} (ID: {inventory_id}) price from ${old_price} to ${new_price}",
                "item": item
            }, indent=2)
    
    available_ids = [item["id"] for item in data.get("items", [])]
    return json.dumps({
        "success": False,
        "error": f"Item with ID '{inventory_id}' not found",
        "available_ids": available_ids
    }, indent=2)

@mcp.tool()
def add_inventory_item(name: str, inventory_id: str, price: str) -> str:
    """
    Add a new item to the inventory
    
    Args:
        name: The name of the inventory item (e.g., "Chocolate")
        inventory_id: The unique ID for the item (e.g., "999")
        price: The price of the item (as a string, e.g., "12.99")
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_inventory()
    
    # Check if ID or name already exists
    for item in data.get("items", []):
        if item["id"] == inventory_id:
            return json.dumps({
                "success": False,
                "error": f"Item with ID '{inventory_id}' already exists",
                "existing_item": item
            }, indent=2)
        if item["name"].lower() == name.lower():
            return json.dumps({
                "success": False,
                "error": f"Item with name '{name}' already exists",
                "existing_item": item
            }, indent=2)
    
    # Add the new item
    new_item = {"id": inventory_id, "name": name, "price": price}
    data.setdefault("items", []).append(new_item)
    save_inventory(data)
    
    return json.dumps({
        "success": True,
        "message": f"Added {name} (ID: {inventory_id}) with price ${price}",
        "item": new_item
    }, indent=2)

@mcp.tool()
def remove_inventory_item(inventory_id: str) -> str:
    """
    Remove an item from the inventory by its ID
    
    Args:
        inventory_id: The ID of the inventory item to remove
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_inventory()
    
    for i, item in enumerate(data.get("items", [])):
        if item["id"] == inventory_id:
            removed_item = data["items"].pop(i)
            save_inventory(data)
            return json.dumps({
                "success": True,
                "message": f"Removed {removed_item['name']} (ID: {inventory_id}, Price: ${removed_item['price']})",
                "removed_item": removed_item
            }, indent=2)
    
    available_ids = [item["id"] for item in data.get("items", [])]
    return json.dumps({
        "success": False,
        "error": f"Item with ID '{inventory_id}' not found",
        "available_ids": available_ids
    }, indent=2)

@mcp.tool()
def update_inventory_item(inventory_id: str, name: str = None, price: str = None) -> str:
    """
    Update any field(s) of an inventory item by its ID
    
    Args:
        inventory_id: The ID of the inventory item to update
        name: The new name (optional, keeps current if not provided)
        price: The new price (optional, keeps current if not provided)
    
    Returns:
        JSON string with confirmation or error message
    """
    data = load_inventory()
    
    for item in data.get("items", []):
        if item["id"] == inventory_id:
            old_item = item.copy()
            
            if name is not None:
                item["name"] = name
            if price is not None:
                item["price"] = price
            
            save_inventory(data)
            return json.dumps({
                "success": True,
                "message": f"Updated item {inventory_id}",
                "old_item": old_item,
                "new_item": item
            }, indent=2)
    
    available_ids = [item["id"] for item in data.get("items", [])]
    return json.dumps({
        "success": False,
        "error": f"Item with ID '{inventory_id}' not found",
        "available_ids": available_ids
    }, indent=2)


if __name__ == "__main__":
    mcp.run()