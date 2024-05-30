PROJECTS
MULTIAGENT AI ASSISTANT
•	Built a multi-agent system for intelligent task completion using LANGCHAIN and OpenAI APIs to leverage GPT-4 for improving productivity and efficiency of users.
•	Implemented 3 specialized agents: Task Planner, Task Executor, and Feedback Agent each designated for specific tasks and these agents are interconnected to form a graph structure for facilitating seamless communication and task execution.
•	Agents are integrated with Memory for storing intermediate results and context, RAG to filter and provide only relevant information to the LLM's from the tools, preventing hallucinations and priming of LLM from the irrelevant context from the tools and necessary tools for enhancing variety of tasks completion accurately and efficiently.
•	Integrated real-time tools to prevent hallucinations and ensure agents operate based on accurate and current information, enhancing reliability and task completion quality. The project leverages LLM's reasoning and capabilities by 50%, enhancing the agents' ability to understand and complete complex tasks effectively.
FINE TUNING LLAMA2 USING QLORA
•	Finetuning the base LLAMA2 using QLORA method with significantly fewer trainable parameters(<1%) to achieve faster training on limited hardware resources (single GPU). Finetuning is done with python code datasets for chat completions of coding related tasks.
•	The base LLAMA2 model is quantized to 4 bit Normal Float using bitsandbytes library and then added the low rank adapter(LORA) layer to linear layers of model using PEFT library for finetuning.
•	Despite quantizing the model and introducing LoRA layers, model reports only a 10% drop in performance compared to the original full-precision model.
TRANSLATOR WITH TRANSFOMER ARCHITECTURE
•	English to Telugu translator with complete transformer architecture built from scratch using keras subclassing APIs for flexibility and control over the architecture's design and customization.
•	Trained the model with custom training loop and scheduled learning rate for faster convergence. Used kerastuner library for best selection among the range of hyperparameter models using custom metric. The model's translation quality is evaluated using BLEU score metric.
•	This architecture with multihead attention significantly improves the model's ability to handle long-term dependencies between words, with enhanced contextual representation and capturing subtle nuances and semantic relationships in text compared to traditional bidirectional LSTMs leading to more accurate and contextually rich translations.
Technical Skills:
•	Programming Language: Python, C++
•	Data Preprocessing and Visualization: SQL, Pandas, Matplotlib, Seaborn
•	ML/DL libraries and frameworks: Keras, TensorFlow, Scikit Learn, Numpy
•	NLP Libraries: spaCy, NLTK, GENISIM, Transformers, PEFT, Langchain, LlamaIndex
•	Versioning Tools: Git, mlflow, DVC
•	Others: Streamlit


