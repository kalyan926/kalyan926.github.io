
<html>
<head>
    <title>My Website</title>
    <style>
         body {
            background-color: #FCFCF9; 
            margin: 0;
            padding: 0;
            align-items: center;
            justify-content: center;
            font-family: Arial, sans-serif;
        }
       .button-container {
            display: flex;
        }
        .button-container button {
            margin: 0;
            padding: 10px 20px;
            font-weight: 500;
            border: 1px solid #ccc;
            background-color: #C9C9C5;
            cursor: pointer;
        }
        .button-container button:hover {
            background-color: #969693;
        }
    </style>
</head>
<body>

<div class="button-container">
    <button onclick="scrollToSection('About me')">About me</button>
    <button onclick="scrollToSection('Technical skills')">Technical skills</button>
    <button onclick="scrollToSection('Projects')">Projects</button>
</div>

<section id="About me">
    <h2>👤 About Me </h2>
    <div style="display: flex; align-items: center;">
  <img src="images/profile_pic.png" alt="Alt text" style="margin-right: 10px; width: 100px;">
  <p> <br>
I am an NLP Data Science enthusiast with expertise in developing custom data-driven models across various domains,Fine-Tuning models using PEFT methods, leveraging large language model(LLM) capabilities with Langchain, creating multi-agent systems with Langchain for intelligent assistance, and building conversational chatbots.</p>
</div>

</section>

<section id="Technical skills">
    <h2>🛠️ Technical Skills </h2>
<p>
    <ul>
    <li> <strong>Programming Languages:</strong> Python, C++ </li>
 <li>   <strong>Data Preprocessing and Visualization:</strong> SQL, Pandas, Matplotlib, Seaborn </li>
<li> <strong>ML/DL Libraries and Frameworks:</strong> Keras, TensorFlow, Scikit Learn, Numpy </li>
<li> <strong>NLP Libraries:</strong> spaCy, NLTK, Gensim, Transformers, PEFT, Langchain, LlamaIndex </li>
<li> <strong>Versioning Tools:</strong> Git, mlflow, DVC </li>
<li> <strong>LLM models:</strong> GPT, LLAMA, BERT, SBERT </li> 
<li> <strong>Vector databases:</strong> Pinecone, ChromaDB </li> 
<li> <strong>Evaluation Frameworks:</strong> DeepEval, Ragas </li>        
<li> <strong>Deployment:</strong> FastAPI, AWS, MCP Server </li> 
<li> <strong>Others:</strong> Streamlit </li></ul> </p> </section>

<section id="Projects">
    <h2>🚀 Projects</h2>

</section>

<script>
    function scrollToSection(id) {
        document.getElementById(id).scrollIntoView({ behavior: 'auto' });
    }
</script>

</body>
</html>


### TRANSLATOR WITH TRANSFOMER ARCHITECTURE [(Project link)](https://github.com/kalyan926/Translator)

- English to Telugu translator with complete **Transformer architecture** built from scratch using **keras subclassing APIs** for flexibility and control over the architecture's design and customization.
- Trained the model with custom training loop and scheduled learning rate for faster convergence. Used **kerastuner** library for best selection among the range of hyperparameter models using custom metric. The model's translation quality is evaluated using **BLEU** score metric.
- The Transformer architecture with multihead attention greatly improves the model's capability to manage long-term dependencies between words, offering improved contextual representation, capturing subtle nuances and semantic relationships crucial for translation and evaluated through **BLEU** score testing.


### INFERENCE TIME COMPUTE REASONING [(Project link)](https://github.com/kalyan926/Inference-Time-Compute-Reasoning)

- Designed and implemented an inference-time reasoning framework for **LLM’s** that simulates **human-like reasoning** using search-based algorithms **(BFS/DFS)**. This approach performs **step-wise thinking** by sampling intermediate steps and making decisions through **evaluation**, allowing intelligent **exploration** of the solution space while leveraging the LLM’s pretrained knowledge for problem solving, achieving a **30–40% improvement in reasoning accuracy** on math and coding tasks.
- **Achieved considerable reasoning performance without costly RLHF/RLVF training**, reducing training compute costs by **~80%**, by utilizing **controlled sampling** and evaluation at inference-time to **guide solution discovery** using existing LLM’s knowledge.

  
### 🧠 MULTIAGENT AI ASSISTANT [(Project link)](https://github.com/kalyan926/MutiAgent-AI-Assistant)

- **Developed** a Multi-Agent System from scratch using **Python** and **Groq APIs**, integrating open-source **LLMs** and the **Tree of Thoughts** technique to enable intelligent, structured task execution. Designed three specialized agents—**Planning**, **Execution**, and  **Evaluator**—coordinated by an **Orchestrator** to ensure seamless communication and workflow. The Execution Agent applies the **Tree of Thoughts** approach to intelligently retry, backtrack, or replan based on dynamic feedback.
- **Enhanced** multi-agent performance by integrating **Memory** (short- and long-term) for contextual continuity, **Retrieval-Augmented Generation (RAG)** for domain-specific knowledge to **reduce hallucinations**, and **external tools** for real-time interaction with the environment. This architecture led to a **50% increase in task completion accuracy**, demonstrating robustness and efficiency in complex task handling.

### AI ASSISTANT END TO END [(Project link)](https://github.com/kalyan926/End-to-End-AI-Assistant)

- Developed an **LLM** by fine-tuning **LLaMA-3.2-1B-Instruct** on synthetic **behavioral ReAct prompt data**, enabling it to intelligently decide between direct answering, database querying, or invoking real-time tools, while reducing input tokens by **60%** and compute overhead by **25%** without compromising performance.
- Optimized model deployment by **quantizing** and converting the fine-tuned model to **GGUF** format using **llama-cpp**, and deployed it on **AWS** cloud. Achieved **12 tokens/sec** inference speed and a **50% boost** in processing throughput, significantly improving realtime responsiveness and production scalability.

### FINE TUNING LLAMA2 USING QLORA [(Project link)](https://github.com/kalyan926/FineTuning-using-QLORA)

- Finetuning the base **LLAMA2-7b** using **QLORA** method with significantly fewer trainable parameters(<1%) to achieve faster training on limited hardware resources (single GPU). Finetuning is done with python code datasets for chat completions of coding related tasks.
- The base LLAMA2 model is quantized to 4 bit Normal Float using bitsandbytes library and then added the low rank adapter(**LORA**) layer to linear layers of model using **PEFT** library for fine tuning. Effectiveness of model is evaluated using **CODEBLEU** metric.
- Fine-tuning with the QLORA technique achieved 90% performance of the original full-precision trained model, using limited resources (a single GPU).




