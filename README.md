
<html>
<head>
    <title>My Website</title>
    <style>
        /* Add some basic styling */
        body {
            font-family: Arial, sans-serif;
        }
        .button-container {
            margin-bottom: 20px;
        }
        button {
            margin-right: 10px;
            padding: 10px 20px;
            cursor: pointer;
        }
        section {
            margin-bottom: 50px;
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
  <strong>Programming Languages:</strong><br>
  - Python<br>
  - C++<br>
  <strong>Data Preprocessing and Visualization:</strong><br>
  - SQL, Pandas, Matplotlib, Seaborn<br>
  <strong>ML/DL Libraries and Frameworks:</strong><br>
  - Keras, TensorFlow, Scikit Learn, Numpy<br>
  <strong>NLP Libraries:</strong><br>
  - spaCy, NLTK, Gensim, Transformers, PEFT, Langchain, LlamaIndex<br>
  <strong>Versioning Tools:</strong><br>
  - Git, mlflow, DVC<br>
  <strong>Others:</strong><br>
  - Streamlit<br>
</p>
</section>

<section id="Projects">
    <h2>🚀 Projects</h2>
<p>### 🧠 Multiagent AI Assistant<br>
- <strong>Overview:</strong> Built a multi-agent system for intelligent task completion using LANGCHAIN and OpenAI APIs to leverage GPT-4.<br>
- <strong>Specialized Agents:</strong> Implemented three specialized agents:<br>
  - <strong>Task Planner:</strong> Plans and schedules tasks.<br>
  - <strong>Task Executor:</strong> Executes the planned tasks.<br>
  - <strong>Feedback Agent:</strong> Provides feedback and improvements.<br>
- <strong>Integration:</strong> Agents are interconnected to form a graph structure, enabling seamless communication and task execution.<br>
- <strong>Memory and RAG:</strong> Integrated with memory for context storage and Retrieval-Augmented Generation (RAG) to filter relevant information, preventing hallucinations and priming from irrelevant context.<br>
- <strong>Real-time Tools:</strong> Ensured agents operate based on accurate and current information, enhancing reliability and task quality.<br>
- <strong>Impact:</strong> Leveraged LLM's reasoning capabilities by 50%, significantly enhancing the agents' ability to understand and complete complex tasks.</p>

<p>### 🦙 Fine Tuning LLAMA2 Using QLORA<br>
- <strong>Method:</strong> Fine-tuned the base LLAMA2 using QLORA method with <1% trainable parameters, achieving faster training on limited hardware (single GPU).<br>
- <strong>Quantization:</strong> Quantized the model to 4-bit Normal Float using bitsandbytes library and added LoRA layers using PEFT library.<br>
- <strong>Performance:</strong> Despite quantization and LoRA layers, the model reported only a 10% drop in performance compared to the original full-precision model.</p>

<p>### 🌐 Translator with Transformer Architecture<br>
- <strong>Architecture:</strong> Developed an English to Telugu translator using complete transformer architecture built from scratch with Keras subclassing APIs.<br>
- <strong>Training:</strong> Employed custom training loop and scheduled learning rate for faster convergence. Used kerastuner for hyperparameter selection.<br>
- <strong>Evaluation:</strong> Model's translation quality evaluated using BLEU score.<br>
- <strong>Enhancements:</strong> Multihead attention significantly improved handling of long-term dependencies, contextual representation, and capturing semantic nuances, leading to accurate and contextually rich translations.</p>

</section>

<script>
    function scrollToSection(id) {
        document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
    }
</script>

</body>
</html>



<div style="display: flex; align-items: center;">
  <img src="images/profile_pic.png" alt="Alt text" style="margin-right: 10px; width: 100px;">
  <p>## <br>
I am an NLP Data Science enthusiast with expertise in developing custom data-driven models across various domains,Fine-Tuning models using PEFT methods, leveraging large language model(LLM) capabilities with Langchain, creating multi-agent systems with Langchain for intelligent assistance, and building conversational chatbots.</p>
</div>






##

### 🧠 Multiagent AI Assistant
- **Overview**: Built a multi-agent system for intelligent task completion using LANGCHAIN and OpenAI APIs to leverage GPT-4.
- **Specialized Agents**: Implemented three specialized agents:
  - **Task Planner**: Plans and schedules tasks.
  - **Task Executor**: Executes the planned tasks.
  - **Feedback Agent**: Provides feedback and improvements.
- **Integration**: Agents are interconnected to form a graph structure, enabling seamless communication and task execution.
- **Memory and RAG**: Integrated with memory for context storage and Retrieval-Augmented Generation (RAG) to filter relevant information, preventing hallucinations and priming from irrelevant context.
- **Real-time Tools**: Ensured agents operate based on accurate and current information, enhancing reliability and task quality.
- **Impact**: Leveraged LLM's reasoning capabilities by 50%, significantly enhancing the agents' ability to understand and complete complex tasks.

### 🦙 Fine Tuning LLAMA2 Using QLORA [(Project link)](https://github.com/kalyan926/FineTuning-using-QLORA)
- **Method**: Fine-tuned the base LLAMA2 using QLORA method with <1% trainable parameters, achieving faster training on limited hardware (single GPU).
- **Quantization**: Quantized the model to 4-bit Normal Float using bitsandbytes library and added LoRA layers using PEFT library.
- **Performance**: Despite quantization and LoRA layers, the model reported only a 10% drop in performance compared to the original full-precision model.

### 🌐 Translator with Transformer Architecture [(Project link)](https://github.com/kalyan926/Translator)
- **Architecture**: Developed an English to Telugu translator using complete transformer architecture built from scratch with Keras subclassing APIs.
- **Training**: Employed custom training loop and scheduled learning rate for faster convergence. Used kerastuner for hyperparameter selection.
- **Evaluation**: Model's translation quality evaluated using BLEU score.
- **Enhancements**: Multihead attention significantly improved handling of long-term dependencies, contextual representation, and capturing semantic nuances, leading to accurate and contextually rich translations.



