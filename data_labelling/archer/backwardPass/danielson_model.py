"""
This module adapts the Danielson evaluation framework into the Model class structure.

It demonstrates how to structure an existing code functionality within the Model
architecture for optimization with AdaLflow.
"""

import os
import sys
from typing import Dict, Any, List, Optional

# Add the parent directory to sys.path to allow for imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from archer.backwardPass.model import Model
from archer.helpers.prompt import Prompt
from eval.danielson import generate_ai_content, normalize_score_integer

class DanielsonModel(Model):
    """
    A Model implementation for the Danielson evaluation framework.
    
    This class wraps the existing Danielson evaluation functionality within
    the Model architecture, making its prompts optimizable through AdaLflow.
    """
    
    def __init__(self, 
                 name: str = "danielson",
                 adalflow_enabled: bool = False,
                 model_type: str = "evaluator",
                 version: str = "1.0.0",
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a new DanielsonModel.
        
        Args:
            name: Name of the model (default: "danielson").
            adalflow_enabled: Whether AdaLflow is enabled for this model.
            model_type: Type of the model (default: "evaluator").
            version: Version of the model.
            metadata: Additional metadata about the model.
        """
        super().__init__(
            name=name,
            model_type=model_type,
            adalflow_enabled=adalflow_enabled,
            version=version,
            metadata=metadata or {}
        )
        
        # Initialize Danielson-specific prompts
        self._initialize_danielson_prompts()
        
        # Register Danielson functions
        self._register_danielson_functions()
    
    def _initialize_danielson_prompts(self):
        """Initialize the prompts used in the Danielson framework."""
        # Context analysis prompt
        self.add_prompt(
            "context_analysis",
            Prompt(
                content="""
                You are an expert in the Charlotte Danielson Framework for Teaching. Analyze the following classroom observation text with a focus on key evaluation aspects.

                ### Your analysis should cover:
                1. **The Four Domains**:
                - Planning and Preparation
                - Classroom Environment
                - Instruction
                - Professional Responsibilities
                2. **The 16 Components** within these domains.
                3. **Performance Levels**: Evaluate and classify observations as Unsatisfactory, Basic, Proficient, or Distinguished.

                ### Key Aspects to Analyze:
                - **Time in the Classroom**: Was the evaluator present for the beginning, middle, end, or the entire session? If unclear, make reasonable inferences.
                - **Evaluation Type**: Identify whether this is an ECE (Early Childhood Education), Special Education, or Standard Evaluation. Use explicit notes or context clues. In the case of each, instruct the evaluator to use the appropriate Danielson rubric. 
                - **Curriculum & Instructional Practices**: Deduce the general grade level and subject based on provided content. Suggest best pedagogical practices aligned with the curriculum.

                ### Your Task:
                1. **Framework-Aligned Evaluation**: Break down how the observation text aligns with each domain and component of the Danielson Framework.
                2. **Evidence Collection**: Extract specific **word-for-word** quotations from the observation text (2-3 impactful examples per component) to support your assessment.
                3. **Missing or Unclear Evidence**: Highlight areas where documentation is insufficient or ambiguous.
                4. **Inferential Analysis**: Where explicit details are missing, use observable behaviors to infer instructional effectiveness. For instance:
                - If students are following directions efficiently, infer strong classroom management.
                - If engagement is high, infer effective instructional strategies.
                5. **Final Summary**: Provide a structured evaluation that includes both performance analysis and potential growth areas for each component.

                **IMPORTANT:**  
                - **Quote directly from the observation text.** Your claims must be backed by specific examples.  
                - **Be explicit in distinguishing direct evidence from inferences.**  
                - **Structure responses clearly by component, using bullet points or numbered lists for clarity.**
                - **Balance detail for administrators with actionable insights that could inform coaching conversations.**

                **Observation Text to Analyze:**
                {text}

                Deliver a structured and detailed evaluation that can be used as context for an official Danielson evaluation.
                """
            )
        )
        
        # Component evaluation base prompt
        self.add_prompt(
            "component_evaluation_base",
            Prompt(
                content="""
                You are an expert evaluator using the Charlotte Danielson Framework for Teaching. 
                Evaluate the teacher observation strictly for component {component_id}. 
                {specific_instruction}

                Context:
                {context}

                Observation:
                {observation_text}

                Guidelines:
                1. Provide a JSON output with two keys: 'summary' and 'score'.
                2. 'score' must be an integer between 1 (Unsatisfactory) and 4 (Distinguished).
                3. 'summary' should include both:
                   - Performance analysis: What the teacher did well with specific evidence
                   - Growth path: 1-3 specific, actionable recommendations for improvement
                4. For each statement in your 'summary', include direct evidence from the observation.
                5. Do not include any commentary outside the JSON structure.
                """
            )
        )
        
        # Restructure feedback prompt
        self.add_prompt(
            "restructure_feedback",
            Prompt(
                content="""
                Please analyze this teacher observation for component {component_id} and create detailed, evidence-based feedback that clearly separates performance analysis from growth opportunities. Model your response after these exemplar evaluations:

                Original Feedback:
                {text}

                Original Evidence/Low Inference Notes:
                {evidence}

                ## Structure your response in exactly this format:

                **Performance Analysis**
                - Begin with a clear statement connecting the teacher's overall performance to their score level for this component
                - Include 2-3 direct quotes from the low inference notes as specific evidence of key strengths or areas of concern
                - Identify 1-2 specific practices that positively impacted student learning
                - Note any missing high-leverage practices that could have elevated their performance
                - This section should be comprehensive enough for administrators while still being concise

                \n
                \n
                
                **Growth Path**
                - Provide 1-3 specific, high-leverage action steps that would have the greatest impact on student learning
                - For each recommendation:
                  - Describe exactly what the teacher should do differently (be specific and actionable)
                  - Explain how this change will improve student learning outcomes
                  - When possible, suggest concrete implementation strategies that could be part of a 6-8 week coaching plan
                  - Include SMART goals or clear metrics to track progress
                - Connect recommendations to observable student behaviors or learning outcomes
                - Keep this section focused, practical and implementable

                ## Important Guidelines:
                - Ground all feedback in specific evidence from the low inference notes using direct quotes
                - Focus on student learning impact rather than just teacher actions
                - Maintain a constructive, growth-oriented tone
                - Be specific and actionable in improvement suggestions
                - Use professional language while remaining accessible
                - Consider the real-world context and practicality of suggestions
                - Reference patterns from exemplar evaluations where appropriate
                - Ensure feedback is framework-aligned and references specific Danielson expectations for this component

                Example Structure:
                "The teacher demonstrated [overall performance level] as evidenced by [specific quoted observation]. Their use of [specific practice] effectively supported student learning by [impact]. While these practices were strong, incorporating [missing element] would have further enhanced student understanding.

                To strengthen their practice, the teacher should consider implementing [specific strategy]. This could be accomplished by [concrete action steps] which would lead to [specific student learning outcome]. Additionally, [second recommendation] would help students [learning impact]."

                You should output it in Markdown format.
                Always start with **Performance Analysis** and then **Growth Path**. Always separate the sections with a new line; they are their own paragraphs.

                Your longer, improved component summary:
                """
            )
        )
        
        # Component-specific instruction prompts
        component_instructions = {
            "1a": "Analyze the clarity and specificity of the lesson objectives, ensuring they are measurable and aligned with both curriculum standards and the teacher's stated goals. Evaluate whether the lesson plan demonstrates a logical progression of activities, anticipates potential challenges, and incorporates differentiated strategies to address diverse student needs. Include 1-2 specific, high-leverage action steps for improvement.",
            "1b": "Examine the accuracy and relevance of the content presented. Assess how well the material is organized to ensure key concepts are introduced in a logical sequence. Look for effective integration of varied resources, including textbooks, technology, or supplementary materials that enhance the lesson's quality. Identify 1-2 actionable strategies to strengthen content knowledge application.",
            "1c": "Evaluate the diversity and effectiveness of the instructional strategies used. Look for evidence that the teacher employs multiple approaches to engage learners. Assess whether the teacher differentiates instruction to address various learning styles and abilities, promoting deep understanding among all students. Suggest 1-2 concrete ways to enhance instructional outcomes through refined strategies.",
            "1d": "Assess the quality of classroom interactions and the level of student engagement. Evaluate how the teacher fosters a positive, inclusive atmosphere through effective communication, active listening, and responsive feedback. Look for evidence of interactive discussions, group collaboration, and techniques that promote critical thinking. Recommend 1-2 specific approaches to deepen student engagement.",
            "1e": "Examine the classroom management strategies and the efficiency of transitions between activities. Evaluate whether the teacher maintains a structured environment with clear routines and procedures that minimize disruptions. Look for smooth transitions that maximize instructional time and support a focused learning atmosphere. Provide 1-2 high-impact suggestions for improving classroom management.",
            "1f": "Review the integration of assessment practices within the lesson. Assess how both formative and summative assessments are used to gauge student understanding. Look for timely, specific feedback that not only informs the students about their progress but also guides the teacher's ongoing instructional adjustments. Recommend 1-2 actionable assessment strategies to implement.",
            "2a": "Evaluate how the teacher establishes and maintains a respectful, productive classroom culture. Look for strategies that promote inclusivity, set clear expectations, and foster mutual respect among students. Assess the proactive measures taken to build a collaborative and supportive learning environment. Suggest 1-2 concrete ways to enhance classroom culture.",
            "2b": "Focus on the physical layout and resource availability within the classroom. Assess whether the arrangement of the space supports both individual and group learning activities. Evaluate the accessibility and organization of materials and resources that enhance the overall learning experience. Recommend 1-2 specific adjustments to optimize the learning environment.",
            "2c": "Assess the clarity and consistency of behavioral expectations communicated by the teacher. Look for well-defined routines and consistent enforcement of rules. Evaluate how the teacher's management strategies support a safe, orderly environment that is fair and conducive to learning. Provide 1-2 high-leverage suggestions for strengthening classroom procedures.",
            "2d": "Examine the teacher's use of proactive interventions to address potential academic or behavioral challenges. Look for evidence of targeted support for individual students, including differentiated interventions and timely adjustments based on ongoing assessments of student needs. Recommend 1-2 specific strategies to enhance student behavior management.",
            "2e": "Consider how both the physical and psychological aspects of the classroom environment impact learning. Assess elements such as lighting, seating, displays, and overall ambiance. Evaluate whether the space is intentionally organized to promote focus, comfort, and engagement. Suggest 1-2 actionable modifications to improve the physical environment.",
            "3a": "Evaluate the clarity and effectiveness of instructional delivery. Examine the teacher's communication style, pacing, and ability to present complex ideas in an accessible manner. Look for a balanced approach that combines direct instruction with interactive, student-centered activities. Provide 1-2 specific techniques to enhance communication effectiveness.",
            "3b": "Look for evidence of effective questioning techniques that stimulate higher-order thinking. Assess whether the teacher employs open-ended questions and probing prompts that encourage analysis, synthesis, and problem-solving. Evaluate how these methods foster student reflection and active engagement in the learning process. Recommend 1-2 questioning strategies to implement.",
            "3c": "Assess the teacher's adaptability in response to the diverse learning needs of students. Examine the use of real-time feedback, flexible grouping, and differentiated instructional strategies. Evaluate whether the teacher uses formative assessment data to make adjustments that enhance understanding for all learners. Suggest 1-2 specific approaches to improve responsiveness to student needs.",
            "3d": "Examine the integration of technology and multimedia resources into the lesson. Assess how digital tools and visual aids are used to enrich the instructional experience. Evaluate whether these resources are relevant to the lesson objectives and effectively enhance student engagement and comprehension. Recommend 1-2 actionable strategies for technology integration.",
            "3e": "Review the overall coherence of the lesson's structure and instructional materials. Assess whether the lesson flows logically, with each segment building upon previous knowledge. Evaluate the alignment of resources, activities, and assessments with the stated learning objectives to ensure a unified and effective learning experience. Provide 1-2 specific suggestions to strengthen lesson coherence."
        }
        
        # Add component-specific instruction prompts
        for component_id, instruction in component_instructions.items():
            self.add_prompt(
                f"component_instruction_{component_id}",
                Prompt(content=instruction)
            )
    
    def _register_danielson_functions(self):
        """Register the functions used in the Danielson framework."""
        self.add_function("analyze_context", self.analyze_danielson_context)
        self.add_function("generate_component_evaluation", self.generate_component_evaluation)
        self.add_function("restructure_feedback", self.restructure_component_feedback)
        self.add_function("generate_single_evaluation", self.generate_single_component_evaluation)
    
    def analyze_danielson_context(self, text: str, model=None) -> Dict[str, Any]:
        """
        Preprocess the observation text by analyzing it in the context of the Danielson Framework.
        
        Args:
            text (str): The evaluation notes text
            model: The model instance (optional, used for function signature compatibility)
            
        Returns:
            Dict: The analysis result and any potential errors
        """
        if model is None:
            model = self
            
        prompt_template = model.get_prompt("context_analysis").content
        prompt = prompt_template.format(text=text)
        
        try:
            
            response = generate_ai_content(prompt)
            if hasattr(response, 'parts') and response.parts:
                analysis_text = ''.join(part.text for part in response.parts)
                return {"analysis": analysis_text, "error": None}
            else:
                return {"analysis": "", "error": "No content generated"}
        except Exception as e:
            return {"analysis": "", "error": str(e)}
    
    def generate_component_evaluation(self, component_id: str, observation_text: str, 
                                     context: str, model=None) -> Dict[str, Any]:
        """
        Generate an evaluation for a specific Danielson component.
        
        Args:
            component_id (str): The Danielson component ID.
            observation_text (str): The observation text.
            context (str): Additional contextual analysis.
            model: The model instance (optional)
            
        Returns:
            Dict[str, Any]: JSON-formatted evaluation with keys 'summary' and 'score'.
        """
        if model is None:
            model = self
            
        # Get the component-specific instruction
        component_instruction_prompt = model.get_prompt(f"component_instruction_{component_id}")
        if component_instruction_prompt:
            specific_instruction = component_instruction_prompt.content
        else:
            specific_instruction = "Focus on key evidence directly related to this component."
        
        # Build the prompt using the base template and specific instruction
        base_prompt = model.get_prompt("component_evaluation_base").content
        prompt = base_prompt.format(
            component_id=component_id,
            specific_instruction=specific_instruction,
            context=context,
            observation_text=observation_text
        )
        
        try:
            import json
            import google.generativeai as genai
            
            response = generate_ai_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    response_mime_type="application/json"
                )
            )
            
            if hasattr(response, 'parts') and response.parts:
                response_text = ''.join(part.text for part in response.parts).strip()
                result = json.loads(response_text)
                result["score"] = normalize_score_integer(result.get("score", 1))
                return result
            else:
                return {"score": 1, "summary": ""}
        except Exception as e:
            return {"score": 1, "summary": f"Error: {str(e)}"}
    
    def restructure_component_feedback(self, text: str, evidence: str, 
                                      component_id: str, model=None) -> str:
        """
        Restructure feedback for a single component using AI.
        
        Args:
            text (str): Original feedback text
            evidence (str): Original observation text containing evidence
            component_id (str): Component identifier (e.g., "1a")
            model: The model instance (optional)
            
        Returns:
            str: Restructured feedback
        """
        if model is None:
            model = self
            
        # Get the restructure feedback prompt template
        prompt_template = model.get_prompt("restructure_feedback").content
        
        # Format the prompt with the inputs
        prompt = prompt_template.format(
            component_id=component_id,
            text=text,
            evidence=evidence
        )
        
        try:
            
            response = generate_ai_content(prompt)
            if hasattr(response, 'parts') and response.parts:
                return ''.join(part.text for part in response.parts)
            else:
                return text  # Return original text if processing fails
        except Exception as e:
            return text  # Return original text if processing fails
    
    def generate_single_component_evaluation(self, low_inference_notes: str, 
                                           component_id: str, model=None) -> Dict[str, Any]:
        """
        Generate an evaluation for a single Danielson Framework component based on low inference notes.
        
        Args:
            low_inference_notes (str): The observation text/low inference notes
            component_id (str): The Danielson component ID (e.g., "1a", "2c", "3e")
            model: The model instance (optional)
            
        Returns:
            Dict[str, Any]: Component evaluation with enhanced feedback
        """
        if model is None:
            model = self
            
        # Step 1: Validate component ID format
        if not (isinstance(component_id, str) and 
                len(component_id) == 2 and 
                component_id[0] in "123" and 
                component_id[1] in "abcdef"):
            return {"error": f"Invalid component ID: {component_id}. Must be in format like '1a', '2c', '3e'"}
        
        # Step 2: Generate contextual analysis for better evaluation
        context_result = model.analyze_danielson_context(low_inference_notes, model)
        
        if context_result.get("error"):
            return {"error": context_result["error"]}
        
        # Step 3: Generate evaluation for the specific component
        component_eval = model.generate_component_evaluation(
            component_id=component_id,
            observation_text=low_inference_notes,
            context=context_result["analysis"],
            model=model
        )
        
        # Step 4: Enhance the feedback with more detailed, actionable information
        enhanced_feedback = model.restructure_component_feedback(
            text=component_eval.get("summary", ""),
            evidence=low_inference_notes,
            component_id=component_id,
            model=model
        )
        
        # Step 5: Assemble final component evaluation
        
        result = {
            "component_id": component_id,
            "score": normalize_score_integer(component_eval.get("score", 1)),
            "summary": enhanced_feedback,
            "domain": component_id[0],  # Extract domain from component ID (first character)
        }
        
        return result 