# 💬 ChatGPT-2024 Style UI - Quick Start

## 🚀 Quick Start (30 seconds)

```bash
# Terminal 1: Backend
cd /Users/gokul/Documents/MARK
python3.10 -m uvicorn src.api.main:app --reload

# Terminal 2: Frontend
cd /Users/gokul/Documents/MARK/ui
npm run dev
```

Open: `http://localhost:5173`

---

## 🎯 What's New

### Before (Old UI)
- Basic text input/output
- No markdown support
- No animations
- No dark mode
- No chat persistence

### After (ChatGPT-2024)
✅ **Sidebar** with chat sessions  
✅ **Markdown** rendering with code highlighting  
✅ **Dark/Light mode** toggle  
✅ **Animations** everywhere (framer-motion)  
✅ **Auto-resize** textarea  
✅ **Message actions** (copy on hover)  
✅ **LocalStorage** persistence  
✅ **Typing indicator** animation  

---

## 📦 Features

### 1. Sidebar Navigation
- Create new chats
- Switch between conversations
- Auto-saves all history

### 2. Message Rendering
- **Markdown**: Headings, lists, tables, blockquotes
- **Code Blocks**: Syntax highlighting for all languages
- **Inline Code**: Styled like ChatGPT
- **Links**: Clickable and styled
- **Bold/Italic**: Full formatting support

### 3. Dark Mode
- Toggle button in sidebar
- Persists across sessions
- Smooth color transitions

### 4. Animations
- Message fade-in
- Typing dots (3 bouncing dots)
- Sidebar slide
- Hover effects

### 5. Smart Textarea
- Auto-expands as you type
- Max height: 160px
- Enter to send
- Shift+Enter for new line

---

## 🎨 View Modes

**Top-right switcher:**
- 💬 **Chat** - ChatGPT interface (NEW!)
- 📊 **Dashboard** - Your 3-column dashboard
- ⚙️ **Admin** - System monitoring

---

## 🔧 Technical Details

### Stack
- **React** - UI framework
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **React Markdown** - Markdown parsing
- **React Syntax Highlighter** - Code highlighting
- **Lucide Icons** - Icon set

### Data Flow
```
User Input
    ↓
ChatGPT Component
    ↓
queryAPI({ query, model: 'auto', top_k: 5 })
    ↓
POST /query → Backend
    ↓
Response with answer
    ↓
Render with Markdown
    ↓
Save to localStorage
```

### Storage
```javascript
// localStorage keys
chatSessions: Array<Session>  // All chat history
darkMode: boolean             // Theme preference
```

---

## 📝 Usage Examples

### Ask a Question
1. Type in the input box
2. Press Enter (or click Send)
3. Watch typing animation
4. See formatted response with markdown

### Switch Chats
1. Click chat session in sidebar
2. History loads instantly
3. Continue conversation

### Copy Response
1. Hover over assistant message
2. Click "Copy" button
3. Text copied to clipboard

### Toggle Dark Mode
1. Click Sun/Moon icon in sidebar
2. Theme switches instantly
3. Preference saved automatically

---

## 🎯 Markdown Support

The UI renders full markdown including:

```markdown
# Heading 1
## Heading 2

**Bold text** and *italic text*

- Bullet list
- Another item

1. Numbered list
2. Another item

`inline code`

\`\`\`python
# Code block with syntax highlighting
def hello():
    print("Hello, World!")
\`\`\`

> Blockquote

| Table | Header |
|-------|--------|
| Cell  | Cell   |

[Link](https://example.com)
```

---

## ⚙️ Configuration

### Change API Endpoint
Edit `ui/src/api.js`:
```javascript
const API_BASE_URL = 'http://localhost:8000'
```

### Adjust Textarea Height
Edit `ChatGPT.jsx`:
```javascript
className="... max-h-40"  // Change max-h-40 to max-h-60, etc.
```

### Change Color Scheme
Edit color classes in `ChatGPT.jsx`:
```javascript
// User message: bg-blue-600 → bg-purple-600
// Assistant: bg-green-600 → bg-indigo-600
```

---

## 🐛 Troubleshooting

### "Cannot find module 'react-markdown'"
```bash
cd ui && npm install react-markdown react-syntax-highlighter framer-motion remark-gfm
```

### "Backend not responding"
```bash
# Check backend is running
curl http://localhost:8000/health

# Start backend
python3.10 -m uvicorn src.api.main:app --reload
```

### "Messages not persisting"
- Check browser localStorage (DevTools → Application → Local Storage)
- Clear cache: `localStorage.clear()`

### "Dark mode not working"
- Check Tailwind dark mode is enabled in `tailwind.config.js`
- Should have `darkMode: 'class'`

---

## 📊 Performance Tips

### Optimize for Large Chats
```javascript
// Limit messages displayed
const recentMessages = messages.slice(-50)

// Virtualize long lists
// Use react-window or react-virtual
```

### Clear Old Chats
```javascript
// In browser console
localStorage.removeItem('chatSessions')
```

---

## 🎉 You're Done!

**Your legal AI now has a world-class ChatGPT interface!**

The UI is:
- ✅ Production-ready
- ✅ Fully functional
- ✅ Beautiful and polished
- ✅ Responsive
- ✅ Accessible

**Start chatting!** 🚀
