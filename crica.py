import tkinter as tk
from tkinter import ttk
import threading
import time
import pyautogui

# Configuration
# SAFE_EXIT ensures the program stops if you slam the mouse to the corner of the screen
pyautogui.FAILSAFE = True 

class AutoClickerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Desktop Auto Clicker")
        self.root.geometry("300x250")
        self.root.resizable(False, False)
        
        # Always stay on top so you can see it over other windows
        self.root.attributes('-topmost', True)

        # State
        self.is_clicking = False
        self.click_thread = threading.Thread(target=self.click_loop)
        self.click_thread.daemon = True # Kills thread when app closes
        self.click_thread.start()

        # UI Layout
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Status Label
        self.status_label = ttk.Label(
            main_frame, 
            text="Status: IDLE", 
            foreground="red",
            font=("Helvetica", 12, "bold")
        )
        self.status_label.pack(pady=(0, 10))

        # Instructions
        info_label = ttk.Label(
            main_frame, 
            text="Press button to toggle clicking.\nUse slider to adjust speed.",
            justify="center"
        )
        info_label.pack(pady=(0, 15))

        # Toggle Button
        self.toggle_btn = ttk.Button(
            main_frame, 
            text="START CLICKING", 
            command=self.toggle_clicking
        )
        self.toggle_btn.pack(pady=5, fill=tk.X)

        # Speed Control (Interval in seconds)
        self.speed_val = tk.DoubleVar(value=0.1)
        
        speed_label = ttk.Label(main_frame, text="Click Interval (seconds):")
        speed_label.pack(pady=(15, 5))
        
        # Scale: from 0.01s (fast) to 2.0s (slow)
        self.speed_scale = ttk.Scale(
            main_frame, 
            from_=0.01, 
            to=2.0, 
            orient='horizontal', 
            variable=self.speed_val,
            command=self.update_speed_label
        )
        self.speed_scale.pack(fill=tk.X)

        self.speed_display = ttk.Label(main_frame, text="0.10 s")
        self.speed_display.pack()

    def update_speed_label(self, val):
        self.speed_display.config(text=f"{float(val):.2f} s")

    def toggle_clicking(self):
        if self.is_clicking:
            self.stop_clicking()
        else:
            self.start_clicking()

    def start_clicking(self):
        self.is_clicking = True
        self.status_label.config(text="Status: CLICKING", foreground="green")
        self.toggle_btn.config(text="STOP CLICKING")
        
    def stop_clicking(self):
        self.is_clicking = False
        self.status_label.config(text="Status: IDLE", foreground="red")
        self.toggle_btn.config(text="START CLICKING")

    def click_loop(self):
        """Runs in a separate thread to keep UI responsive."""
        while True:
            if self.is_clicking:
                pyautogui.click()
                # Sleep for the duration set on the slider
                time.sleep(self.speed_val.get())
            else:
                # Sleep briefly to reduce CPU usage when idle
                time.sleep(0.1)

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoClickerApp(root)
    root.mainloop()