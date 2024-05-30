<!DOCTYPE html>
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
    <button onclick="scrollToSection('section1')">Section 1</button>
    <button onclick="scrollToSection('section2')">Section 2</button>
    <button onclick="scrollToSection('section3')">Section 3</button>
</div>

<section id="section1">
    <h2>Section 1</h2>
    <p>This is the content of Section 1. Here you can provide detailed information about this section.</p>
</section>

<section id="section2">
    <h2>Section 2</h2>
    <p>This is the content of Section 2. More detailed information can be added here.</p>
</section>

<section id="section3">
    <h2>Section 3</h2>
    <p>This is the content of Section 3. Include all the necessary details for this section.</p>
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
  <p>## 👤 About Me <br>
I am an NLP Data Science enthusiast with expertise in developing custom data-driven models across various domains,Fine-Tuning models using PEFT methods, leveraging large language model(LLM) capabilities with Langchain, creating multi-agent systems with Langchain for intelligent assistance, and building conversational chatbots.</p>
</div>




## 🛠️ Technical Skills

- **Programming Languages**: Python, C++
- **Data Preprocessing and Visualization**: SQL, Pandas, Matplotlib, Seaborn
- **ML/DL Libraries and Frameworks**: Keras, TensorFlow, Scikit Learn, Numpy
- **NLP Libraries**: spaCy, NLTK, Gensim, Transformers, PEFT, Langchain, LlamaIndex
- **Versioning Tools**: Git, mlflow, DVC
- **Others**: Streamlit

## 🚀 Projects

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



