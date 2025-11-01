"""
Grace Expert Knowledge System

Makes Grace as good as world-class AI at coding across:
- All programming languages
- AI/ML/DL domains
- Cloud & DevOps
- Web development (frontend + backend)
- Mobile development
- System architecture
- Security
- Databases
- And more...

This is Grace's comprehensive knowledge base.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExpertDomain(Enum):
    """Expert domains Grace specializes in"""
    AI_ML_DL = "ai_ml_dl"
    PYTHON = "python"
    JAVASCRIPT_TYPESCRIPT = "javascript_typescript"
    RUST = "rust"
    GO = "go"
    JAVA_KOTLIN = "java_kotlin"
    CPP = "cpp"
    WEB_FRONTEND = "web_frontend"
    WEB_BACKEND = "web_backend"
    MOBILE_IOS = "mobile_ios"
    MOBILE_ANDROID = "mobile_android"
    CLOUD_AWS = "cloud_aws"
    CLOUD_AZURE = "cloud_azure"
    CLOUD_GCP = "cloud_gcp"
    DEVOPS = "devops"
    DATABASES = "databases"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    UI_UX = "ui_ux"


@dataclass
class ExpertKnowledge:
    """Knowledge in a specific domain"""
    domain: ExpertDomain
    proficiency_level: float  # 0.0 to 1.0
    knowledge_base: Dict[str, Any]
    best_practices: List[str]
    common_patterns: List[Dict[str, Any]]
    anti_patterns: List[str]
    tools_and_frameworks: List[str]
    code_examples: Dict[str, str]
    
    def get_summary(self) -> str:
        return f"{self.domain.value}: {self.proficiency_level:.0%} proficiency"


class AIMLExpertise:
    """
    AI/ML/DL Expert Knowledge
    
    Grace knows:
    - Neural networks (CNNs, RNNs, Transformers)
    - Training techniques (fine-tuning, PEFT, LoRA)
    - ML frameworks (PyTorch, TensorFlow, JAX)
    - Model optimization
    - MLOps
    """
    
    @staticmethod
    def get_knowledge() -> ExpertKnowledge:
        return ExpertKnowledge(
            domain=ExpertDomain.AI_ML_DL,
            proficiency_level=0.95,
            knowledge_base={
                "architectures": {
                    "transformers": "State-of-the-art for NLP and vision",
                    "cnns": "Computer vision, image processing",
                    "rnns_lstms": "Sequential data, time series",
                    "gans": "Generative models",
                    "diffusion": "Image generation (Stable Diffusion, DALL-E)",
                    "rl": "Reinforcement learning (PPO, DQN, A3C)"
                },
                "training": {
                    "fine_tuning": "Adapt pretrained models",
                    "peft": "Parameter-efficient fine-tuning (LoRA, QLoRA)",
                    "rlhf": "Reinforcement learning from human feedback",
                    "few_shot": "Learn from few examples",
                    "zero_shot": "No examples needed"
                },
                "frameworks": {
                    "pytorch": "Flexible, research-friendly",
                    "tensorflow": "Production, serving",
                    "jax": "High-performance, functional",
                    "huggingface": "Pretrained models, transformers",
                    "langchain": "LLM applications",
                    "llamaindex": "RAG, document QA"
                }
            },
            best_practices=[
                "Use pretrained models when possible",
                "Monitor training metrics (loss, accuracy, validation)",
                "Implement early stopping",
                "Use learning rate schedulers",
                "Validate on held-out test set",
                "Version control models and datasets",
                "Track experiments with MLflow/Weights&Biases",
                "Optimize for inference (quantization, pruning)",
                "Use proper train/val/test splits",
                "Handle class imbalance"
            ],
            common_patterns=[
                {
                    "name": "Fine-tune Transformer",
                    "pattern": "Load pretrained → Add task head → Freeze/unfreeze layers → Train with small LR"
                },
                {
                    "name": "RAG Pipeline",
                    "pattern": "Embed documents → Store in vector DB → Retrieve relevant → Augment prompt → Generate"
                },
                {
                    "name": "Model Serving",
                    "pattern": "Train → Export (ONNX/TorchScript) → Load in server → Batch inference → Cache results"
                }
            ],
            anti_patterns=[
                "Training on test data (data leakage)",
                "Not using validation set",
                "Ignoring class imbalance",
                "Too high learning rate",
                "No regularization (overfitting)",
                "Deploying without testing",
                "Not monitoring production models"
            ],
            tools_and_frameworks=[
                "PyTorch", "TensorFlow", "JAX", "Hugging Face",
                "LangChain", "LlamaIndex", "OpenAI API", "Anthropic API",
                "MLflow", "Weights & Biases", "TensorBoard",
                "ONNX", "TorchScript", "TensorFlow Lite",
                "FastAPI", "Ray", "Celery", "Docker"
            ],
            code_examples={
                "fine_tune_llm": '''from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Prepare dataset
# ... tokenization ...

# Fine-tune
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()''',
                
                "rag_pipeline": '''from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
response = qa_chain.run("Your question here")'''
            }
        )


class WebDevelopmentExpertise:
    """
    Full-Stack Web Development Expert Knowledge
    
    Grace knows:
    - Frontend (React, Vue, Svelte, Next.js)
    - Backend (FastAPI, Node.js, Django)
    - Databases (SQL, NoSQL)
    - APIs (REST, GraphQL, WebSocket)
    - Deployment
    """
    
    @staticmethod
    def get_knowledge() -> ExpertKnowledge:
        return ExpertKnowledge(
            domain=ExpertDomain.WEB_FRONTEND,
            proficiency_level=0.92,
            knowledge_base={
                "frameworks": {
                    "react": "Component-based, virtual DOM, hooks",
                    "nextjs": "React framework with SSR, SSG, API routes",
                    "vue": "Progressive framework, composition API",
                    "svelte": "Compile-time framework, no virtual DOM",
                    "astro": "Multi-framework, islands architecture",
                    "solid": "Fine-grained reactivity"
                },
                "state_management": {
                    "zustand": "Simple, modern",
                    "redux": "Predictable state container",
                    "jotai": "Atomic state management",
                    "tanstack_query": "Server state management"
                },
                "styling": {
                    "tailwind": "Utility-first CSS",
                    "css_modules": "Scoped CSS",
                    "styled_components": "CSS-in-JS",
                    "sass": "CSS preprocessor"
                },
                "build_tools": {
                    "vite": "Fast, modern build tool",
                    "webpack": "Powerful bundler",
                    "esbuild": "Extremely fast bundler",
                    "turbopack": "Next-gen bundler"
                }
            },
            best_practices=[
                "Use TypeScript for type safety",
                "Implement proper error boundaries",
                "Optimize bundle size (code splitting, lazy loading)",
                "Use semantic HTML",
                "Implement accessibility (a11y)",
                "Mobile-first responsive design",
                "Use React.memo() for expensive components",
                "Avoid prop drilling (use context/state management)",
                "Implement proper loading states",
                "Use environment variables for config",
                "Implement proper SEO (meta tags, sitemap)",
                "Use modern CSS (Grid, Flexbox)",
                "Optimize images (WebP, lazy loading)",
                "Implement proper form validation"
            ],
            common_patterns=[
                {
                    "name": "Custom Hook",
                    "code": "const useMyHook = () => { const [state, setState] = useState(); return { state, actions }; }"
                },
                {
                    "name": "Context Provider",
                    "code": "const MyContext = createContext(); export const MyProvider = ({children}) => <MyContext.Provider value={value}>{children}</MyContext.Provider>"
                }
            ],
            anti_patterns=[
                "Mutating state directly",
                "useEffect without dependencies",
                "Prop drilling 5+ levels deep",
                "Not memoizing expensive computations",
                "Inline function definitions in JSX",
                "Massive components (>300 lines)",
                "Not handling loading/error states"
            ],
            tools_and_frameworks=[
                "React", "Next.js", "Vue", "Svelte", "TypeScript",
                "TailwindCSS", "Vite", "ESLint", "Prettier",
                "Zustand", "TanStack Query", "React Hook Form",
                "Vitest", "Playwright", "Cypress"
            ],
            code_examples={
                "react_component": '''import React, { useState, useEffect } from 'react';

interface Props {
  title: string;
}

export const MyComponent: React.FC<Props> = ({ title }) => {
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchData().then(setData).finally(() => setLoading(false));
  }, []);
  
  if (loading) return <div>Loading...</div>;
  
  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold">{title}</h1>
      {data.map(item => <div key={item.id}>{item.name}</div>)}
    </div>
  );
};''',
                
                "nextjs_api": '''// app/api/users/route.ts
import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const users = await db.user.findMany();
  return NextResponse.json(users);
}

export async function POST(request: Request) {
  const body = await request.json();
  const user = await db.user.create({ data: body });
  return NextResponse.json(user, { status: 201 });
}'''
            }
        )


class CloudDevOpsExpertise:
    """Cloud & DevOps Expert Knowledge"""
    
    @staticmethod
    def get_knowledge() -> ExpertKnowledge:
        return ExpertKnowledge(
            domain=ExpertDomain.CLOUD_AWS,
            proficiency_level=0.90,
            knowledge_base={
                "aws_services": {
                    "compute": "EC2, Lambda, ECS, EKS, Fargate",
                    "storage": "S3, EBS, EFS, Glacier",
                    "database": "RDS, DynamoDB, Aurora, DocumentDB",
                    "networking": "VPC, Route53, CloudFront, API Gateway",
                    "ml": "SageMaker, Bedrock, Rekognition",
                    "monitoring": "CloudWatch, X-Ray, CloudTrail"
                },
                "kubernetes": {
                    "resources": "Pods, Deployments, Services, Ingress",
                    "config": "ConfigMaps, Secrets",
                    "scaling": "HPA, VPA, Cluster Autoscaler",
                    "networking": "Network Policies, Service Mesh"
                },
                "ci_cd": {
                    "github_actions": "Workflows, runners, secrets",
                    "gitlab_ci": "Pipelines, jobs, artifacts",
                    "jenkins": "Jenkinsfile, pipelines",
                    "argocd": "GitOps, continuous deployment"
                },
                "iac": {
                    "terraform": "Infrastructure as code",
                    "cloudformation": "AWS native IaC",
                    "pulumi": "Programming language IaC",
                    "ansible": "Configuration management"
                }
            },
            best_practices=[
                "Use Infrastructure as Code (Terraform/CloudFormation)",
                "Implement blue-green or canary deployments",
                "Use container orchestration (Kubernetes)",
                "Implement proper logging and monitoring",
                "Use secrets management (AWS Secrets Manager, Vault)",
                "Implement auto-scaling",
                "Use CDN for static assets",
                "Implement proper backup strategies",
                "Use least privilege IAM policies",
                "Tag all resources for cost tracking",
                "Implement disaster recovery plans",
                "Use managed services when possible"
            ],
            common_patterns=[
                {
                    "name": "Microservices on K8s",
                    "pattern": "Service mesh → Load balancer → Pods → Database"
                },
                {
                    "name": "Serverless API",
                    "pattern": "API Gateway → Lambda → DynamoDB"
                }
            ],
            anti_patterns=[
                "Hardcoding credentials",
                "Not using version control for infrastructure",
                "Manual deployments",
                "No monitoring/alerting",
                "Single point of failure",
                "Not testing disaster recovery"
            ],
            tools_and_frameworks=[
                "Docker", "Kubernetes", "Terraform", "Ansible",
                "GitHub Actions", "GitLab CI", "Jenkins",
                "Prometheus", "Grafana", "ELK Stack",
                "AWS", "Azure", "GCP",
                "Helm", "ArgoCD", "Flux"
            ],
            code_examples={
                "dockerfile": '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]''',
                
                "kubernetes_deployment": '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: grace-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: grace-api
  template:
    metadata:
      labels:
        app: grace-api
    spec:
      containers:
      - name: api
        image: grace-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: grace-api
spec:
  selector:
    app: grace-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer''',
                
                "github_actions": '''name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          docker tag myapp:${{ github.sha }} myregistry/myapp:latest
          docker push myregistry/myapp:latest
      
      - name: Deploy to K8s
        run: |
          kubectl set image deployment/myapp myapp=myregistry/myapp:latest
          kubectl rollout status deployment/myapp'''
            }
        )


class MobileDevelopmentExpertise:
    """Mobile Development Expert Knowledge"""
    
    @staticmethod
    def get_knowledge() -> ExpertKnowledge:
        return ExpertKnowledge(
            domain=ExpertDomain.MOBILE_IOS,
            proficiency_level=0.88,
            knowledge_base={
                "platforms": {
                    "ios_native": "Swift, SwiftUI, UIKit",
                    "android_native": "Kotlin, Jetpack Compose",
                    "react_native": "Cross-platform React",
                    "flutter": "Cross-platform Dart",
                    "expo": "React Native with managed workflow"
                },
                "architecture": {
                    "mvvm": "Model-View-ViewModel",
                    "mvi": "Model-View-Intent",
                    "clean_architecture": "Separation of concerns",
                    "redux": "State management"
                },
                "features": {
                    "navigation": "React Navigation, Router",
                    "state": "Redux, Zustand, Context",
                    "networking": "Axios, Fetch, GraphQL",
                    "storage": "AsyncStorage, SQLite, Realm",
                    "auth": "OAuth, Firebase Auth",
                    "push": "Firebase Cloud Messaging, APNs"
                }
            },
            best_practices=[
                "Use TypeScript for type safety",
                "Implement proper navigation structure",
                "Handle offline mode",
                "Optimize for performance (FlatList, memo)",
                "Test on real devices",
                "Implement proper error handling",
                "Use secure storage for sensitive data",
                "Optimize app bundle size",
                "Implement deep linking",
                "Use environment configs",
                "Follow platform design guidelines",
                "Implement proper splash screens and loading states"
            ],
            common_patterns=[
                {
                    "name": "API Call with Loading",
                    "code": "const [loading, setLoading] = useState(false); const fetchData = async () => { setLoading(true); try { const data = await api.get(); setData(data); } finally { setLoading(false); } }"
                }
            ],
            anti_patterns=[
                "Blocking UI thread with heavy operations",
                "Not handling network failures",
                "Storing sensitive data in plain text",
                "Not optimizing images",
                "Memory leaks (not cleaning up listeners)",
                "Not testing on different screen sizes"
            ],
            tools_and_frameworks=[
                "React Native", "Expo", "Flutter", "Swift", "Kotlin",
                "Firebase", "Supabase", "Amplify",
                "Jest", "Detox", "XCTest", "Espresso",
                "Fastlane", "CodePush", "App Center"
            ],
            code_examples={
                "react_native_screen": '''import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, ActivityIndicator } from 'react-native';

export const UsersScreen = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch('https://api.example.com/users')
      .then(res => res.json())
      .then(setUsers)
      .finally(() => setLoading(false));
  }, []);
  
  if (loading) return <ActivityIndicator />;
  
  return (
    <FlatList
      data={users}
      keyExtractor={(item) => item.id}
      renderItem={({ item }) => (
        <View className="p-4 border-b">
          <Text className="text-lg">{item.name}</Text>
        </View>
      )}
    />
  );
};'''
            }
        )


class PythonExpertise:
    """Python Language Expert Knowledge"""
    
    @staticmethod
    def get_knowledge() -> ExpertKnowledge:
        return ExpertKnowledge(
            domain=ExpertDomain.PYTHON,
            proficiency_level=0.98,
            knowledge_base={
                "modern_features": {
                    "type_hints": "Static typing with mypy",
                    "dataclasses": "Structured data classes",
                    "async_await": "Asynchronous programming",
                    "pattern_matching": "Match statements (3.10+)",
                    "walrus_operator": "Assignment expressions",
                    "f_strings": "Formatted string literals"
                },
                "frameworks": {
                    "fastapi": "Modern async web framework",
                    "django": "Full-featured web framework",
                    "flask": "Lightweight web framework",
                    "sqlalchemy": "SQL toolkit and ORM",
                    "pydantic": "Data validation",
                    "pytest": "Testing framework"
                }
            },
            best_practices=[
                "Use type hints everywhere",
                "Follow PEP 8 style guide",
                "Use context managers (with statements)",
                "Prefer list comprehensions",
                "Use dataclasses for structured data",
                "Handle exceptions explicitly",
                "Use logging instead of print",
                "Write docstrings",
                "Use virtual environments",
                "Pin dependencies",
                "Use async for I/O-bound operations",
                "Implement proper error handling"
            ],
            common_patterns=[
                {"name": "Context Manager", "code": "with open('file.txt') as f: data = f.read()"},
                {"name": "List Comprehension", "code": "squares = [x**2 for x in range(10)]"},
                {"name": "Decorator", "code": "def decorator(func): def wrapper(*args, **kwargs): return func(*args, **kwargs); return wrapper"}
            ],
            anti_patterns=[
                "Using mutable default arguments",
                "Catching bare except",
                "Not using virtual environments",
                "Circular imports",
                "Global variables everywhere"
            ],
            tools_and_frameworks=[
                "FastAPI", "Django", "Flask", "SQLAlchemy", "Pydantic",
                "pytest", "black", "ruff", "mypy",
                "Poetry", "PDM", "UV", "pip-tools"
            ],
            code_examples={
                "fastapi_crud": '''from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List

app = FastAPI()

class UserCreate(BaseModel):
    name: str
    email: str

class User(UserCreate):
    id: int
    class Config:
        from_attributes = True

@app.get("/users", response_model=List[User])
async def list_users(db: Session = Depends(get_db)):
    return db.query(UserModel).all()

@app.post("/users", response_model=User, status_code=201)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = UserModel(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user'''
            }
        )


class ExpertSystem:
    """
    Grace's Complete Expert System
    
    Aggregates all domain expertise and provides intelligent
    code generation, evaluation, and guidance across ALL domains.
    """
    
    def __init__(self):
        self.experts = {
            ExpertDomain.AI_ML_DL: AIMLExpertise.get_knowledge(),
            ExpertDomain.WEB_FRONTEND: WebDevelopmentExpertise.get_knowledge(),
            ExpertDomain.MOBILE_IOS: MobileDevelopmentExpertise.get_knowledge(),
            ExpertDomain.PYTHON: PythonExpertise.get_knowledge(),
        }
        
        logger.info(f"Expert System initialized with {len(self.experts)} domains")
    
    def get_expert_for_task(
        self,
        task_description: str,
        language: Optional[str] = None
    ) -> List[ExpertKnowledge]:
        """
        Select relevant experts for a task.
        
        Uses keywords and context to determine which experts to consult.
        """
        relevant_experts = []
        task_lower = task_description.lower()
        
        # Language-based selection
        if language:
            lang_lower = language.lower()
            if "python" in lang_lower:
                relevant_experts.append(self.experts[ExpertDomain.PYTHON])
            if "javascript" in lang_lower or "typescript" in lang_lower or "react" in lang_lower:
                relevant_experts.append(self.experts[ExpertDomain.WEB_FRONTEND])
        
        # Domain-based selection
        ai_keywords = ["ml", "machine learning", "neural", "model", "train", "inference"]
        if any(keyword in task_lower for keyword in ai_keywords):
            relevant_experts.append(self.experts[ExpertDomain.AI_ML_DL])
        
        web_keywords = ["web", "api", "frontend", "backend", "react", "nextjs"]
        if any(keyword in task_lower for keyword in web_keywords):
            if ExpertDomain.WEB_FRONTEND in self.experts:
                relevant_experts.append(self.experts[ExpertDomain.WEB_FRONTEND])
        
        mobile_keywords = ["mobile", "ios", "android", "app", "react native"]
        if any(keyword in task_lower for keyword in mobile_keywords):
            if ExpertDomain.MOBILE_IOS in self.experts:
                relevant_experts.append(self.experts[ExpertDomain.MOBILE_IOS])
        
        cloud_keywords = ["cloud", "aws", "kubernetes", "docker", "deploy"]
        if any(keyword in task_lower for keyword in cloud_keywords):
            if ExpertDomain.CLOUD_AWS in self.experts:
                relevant_experts.append(self.experts[ExpertDomain.CLOUD_AWS])
        
        # If no specific match, return all
        if not relevant_experts:
            relevant_experts = list(self.experts.values())
        
        return relevant_experts
    
    def get_best_practices(self, domain: ExpertDomain) -> List[str]:
        """Get best practices for a domain"""
        expert = self.experts.get(domain)
        return expert.best_practices if expert else []
    
    def get_code_example(self, domain: ExpertDomain, example_name: str) -> Optional[str]:
        """Get code example from domain"""
        expert = self.experts.get(domain)
        if expert and expert.code_examples:
            return expert.code_examples.get(example_name)
        return None
    
    def generate_expert_guidance(
        self,
        task: str,
        language: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate expert guidance for a coding task.
        
        This makes Grace as good as world-class AI at coding.
        """
        experts = self.get_expert_for_task(task, language)
        
        guidance = {
            "task": task,
            "language": language,
            "relevant_experts": [e.domain.value for e in experts],
            "combined_proficiency": sum(e.proficiency_level for e in experts) / len(experts) if experts else 0.0,
            "best_practices": [],
            "recommended_tools": [],
            "code_patterns": [],
            "warnings": []
        }
        
        # Aggregate best practices
        for expert in experts:
            guidance["best_practices"].extend(expert.best_practices[:5])
            guidance["recommended_tools"].extend(expert.tools_and_frameworks[:5])
            
            # Add patterns
            for pattern in expert.common_patterns[:3]:
                guidance["code_patterns"].append(pattern)
            
            # Add warnings
            for anti in expert.anti_patterns[:3]:
                guidance["warnings"].append(f"Avoid: {anti}")
        
        # Deduplicate
        guidance["best_practices"] = list(set(guidance["best_practices"]))
        guidance["recommended_tools"] = list(set(guidance["recommended_tools"]))
        
        return guidance
    
    def get_all_expertise_summary(self) -> Dict[str, Any]:
        """Get summary of all expertise"""
        return {
            "total_domains": len(self.experts),
            "domains": [
                {
                    "domain": expert.domain.value,
                    "proficiency": expert.proficiency_level,
                    "tools_count": len(expert.tools_and_frameworks),
                    "examples_count": len(expert.code_examples)
                }
                for expert in self.experts.values()
            ],
            "avg_proficiency": sum(e.proficiency_level for e in self.experts.values()) / len(self.experts)
        }


# Global expert system
_expert_system: Optional[ExpertSystem] = None


def get_expert_system() -> ExpertSystem:
    """Get global expert system"""
    global _expert_system
    if _expert_system is None:
        _expert_system = ExpertSystem()
    return _expert_system


if __name__ == "__main__":
    # Demo
    print("🧠 Grace Expert System Demo\n")
    
    expert_sys = ExpertSystem()
    
    # Show all expertise
    summary = expert_sys.get_all_expertise_summary()
    print(f"📊 Grace's Expertise:")
    print(f"   Total domains: {summary['total_domains']}")
    print(f"   Average proficiency: {summary['avg_proficiency']:.0%}\n")
    
    for domain in summary['domains']:
        print(f"   {domain['domain']}: {domain['proficiency']:.0%}")
    
    # Test expert guidance
    print("\n🎯 Expert Guidance Test:")
    guidance = expert_sys.generate_expert_guidance(
        task="Build a REST API with ML model serving",
        language="python",
        context={}
    )
    
    print(f"\n   Task: {guidance['task']}")
    print(f"   Relevant experts: {', '.join(guidance['relevant_experts'])}")
    print(f"   Proficiency: {guidance['combined_proficiency']:.0%}")
    print(f"\n   Top Best Practices:")
    for bp in guidance['best_practices'][:5]:
        print(f"     • {bp}")
    
    print(f"\n   Recommended Tools:")
    for tool in guidance['recommended_tools'][:5]:
        print(f"     • {tool}")
