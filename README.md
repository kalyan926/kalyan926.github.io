
<html>
<head>
    <title>My Website</title>
    <style>
         body {
            background-color: #FCFCF9; 
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }
       .button-container {
            display: flex;
        }
        .button-container button {
            margin: 0;
            padding: 10px 20px;
            border: 1px solid #ccc;
            background-color: #F3F3EE;
            cursor: pointer;
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
<li> <strong>Others:</strong> Streamlit </li>
    </ul> </p> </section>

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



