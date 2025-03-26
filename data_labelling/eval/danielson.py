from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from dotenv import load_dotenv

# Global ThreadPoolExecutor for parallel API calls
executor = ThreadPoolExecutor(max_workers=int(os.getenv("THREADPOOL_MAX_WORKERS", "20")))

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class PreprocessResult(TypedDict):
    analysis: str
    error: Optional[str]

class ComponentFeedback(TypedDict):
    domain: str
    component: str
    summary: str


# wrapper function which creates ONE component evaluation
#  evaluation = generate_single_component_evaluation("Your low inference notes here...", "1a")
# we will certainly want to export this function for creating data points for the hf space
def generate_single_component_evaluation(low_inference_notes: str, component_id: str) -> Dict[str, Any]:
    """
    Generate an evaluation for a single Danielson Framework component based on low inference notes.
    
    Args:
        low_inference_notes (str): The observation text/low inference notes
        component_id (str): The Danielson component ID (e.g., "1a", "2c", "3e")
        
    Returns:
        Dict[str, Any]: Component evaluation with enhanced feedback
    """
    # Step 1: Validate component ID format
    if not (isinstance(component_id, str) and 
            len(component_id) == 2 and 
            component_id[0] in "123" and 
            component_id[1] in "abcdef"):
        return {"error": f"Invalid component ID: {component_id}. Must be in format like '1a', '2c', '3e'"}
    
    # Step 2: Generate contextual analysis for better evaluation
    context_result = analyze_danielson_context(low_inference_notes)
    
    if context_result["error"]:
        return {"error": context_result["error"]}
    
    # Step 3: Generate evaluation for the specific component
    component_eval = generate_component_evaluation(
        component_id=component_id,
        observation_text=low_inference_notes,
        context=context_result["analysis"]
    )
    
    # Step 4: Enhance the feedback with more detailed, actionable information
    enhanced_feedback = restructure_component_feedback(
        text=component_eval.get("summary", ""),
        evidence=low_inference_notes,
        component_id=component_id
    )
    
    # Step 5: Assemble final component evaluation
    result = {
        "component_id": component_id,
        "score": normalize_score_integer(component_eval.get("score", 1)),
        "summary": enhanced_feedback,
        "domain": component_id[0],  # Extract domain from component ID (first character)
    }
    
    return result


def generate_ai_content(prompt: str, generation_config: Optional[Any] = None):
    """
    Wrapper for generating AI content using either Gemini or Groq.
    For Groq, we use the "llama-3.3-70b-versatile" model and map the generation
    parameters (e.g. temperature and JSON output) per the Groq API documentation.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt, generation_config=generation_config)

def analyze_danielson_context(text: str):
    """
    Preprocess the observation text by analyzing it in the context of the Danielson Framework.
    
    Args:
        text (str): The evaluation notes text
        
    Returns:
        PreprocessResult: The analysis result and any potential errors
    """
    analyze_danielson_context_prompt = f"""
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
    
    try:
        response = generate_ai_content(analyze_danielson_context_prompt)
        if response.parts:
            analysis_text = ''.join(part.text for part in response.parts)
            return {"analysis": analysis_text, "error": None}
        else:
            return {"analysis": "", "error": "No content generated"}
    except Exception as e:
        return {"analysis": "", "error": str(e)}

def create_component_prompt(component_id: str, observation_text: str, context: str) -> str:
    """
    Dynamically create a concise prompt for evaluating a specific Danielson Framework component.
    This function integrates:
      - Dynamic prompt generation: reducing overall verbosity by only including essential details.
      - Component-specific instructions: mapping unique evaluation criteria to each component.
    
    Args:
        component_id (str): The Danielson component ID (e.g., "1a", "3b").
        observation_text (str): The teacher observation text.
        context (str): Additional contextual information (e.g., results from pre-analysis).
    
    Returns:
        str: A tailored prompt to be passed to the AI content generator.
    """
    # Base instructions common to all components.
    base_prompt = (
        f"You are an expert evaluator using the Charlotte Danielson Framework for Teaching. "
        f"Evaluate the teacher observation strictly for component {component_id}. "
    )
    
    # Component-specific instructions: these details force the model to focus on unique aspects.
    component_instructions = {
        "1a": (
            "Analyze the clarity and specificity of the lesson objectives, ensuring they are measurable "
            "and aligned with both curriculum standards and the teacher's stated goals. Evaluate whether "
            "the lesson plan demonstrates a logical progression of activities, anticipates potential challenges, "
            "and incorporates differentiated strategies to address diverse student needs. Include 1-2 specific, "
            "high-leverage action steps for improvement."
        ),
        "1b": (
            "Examine the accuracy and relevance of the content presented. Assess how well the material is organized "
            "to ensure key concepts are introduced in a logical sequence. Look for effective integration of varied resources, "
            "including textbooks, technology, or supplementary materials that enhance the lesson's quality. Identify 1-2 "
            "actionable strategies to strengthen content knowledge application."
        ),
        "1c": (
            "Evaluate the diversity and effectiveness of the instructional strategies used. Look for evidence that the teacher "
            "employs multiple approaches to engage learners. Assess whether the teacher differentiates instruction to address various "
            "learning styles and abilities, promoting deep understanding among all students. Suggest 1-2 concrete ways to enhance "
            "instructional outcomes through refined strategies."
        ),
        "1d": (
            "Assess the quality of classroom interactions and the level of student engagement. Evaluate how the teacher fosters a "
            "positive, inclusive atmosphere through effective communication, active listening, and responsive feedback. Look for evidence "
            "of interactive discussions, group collaboration, and techniques that promote critical thinking. Recommend 1-2 specific "
            "approaches to deepen student engagement."
        ),
        "1e": (
            "Examine the classroom management strategies and the efficiency of transitions between activities. Evaluate whether the teacher "
            "maintains a structured environment with clear routines and procedures that minimize disruptions. Look for smooth transitions that maximize "
            "instructional time and support a focused learning atmosphere. Provide 1-2 high-impact suggestions for improving classroom management."
        ),
        "1f": (
            "Review the integration of assessment practices within the lesson. Assess how both formative and summative assessments are used to "
            "gauge student understanding. Look for timely, specific feedback that not only informs the students about their progress but also guides the teacher's "
            "ongoing instructional adjustments. Recommend 1-2 actionable assessment strategies to implement."
        ),
        "2a": (
            "Evaluate how the teacher establishes and maintains a respectful, productive classroom culture. Look for strategies that promote inclusivity, set clear "
            "expectations, and foster mutual respect among students. Assess the proactive measures taken to build a collaborative and supportive learning environment. "
            "Suggest 1-2 concrete ways to enhance classroom culture."
        ),
        "2b": (
            "Focus on the physical layout and resource availability within the classroom. Assess whether the arrangement of the space supports both individual and "
            "group learning activities. Evaluate the accessibility and organization of materials and resources that enhance the overall learning experience. "
            "Recommend 1-2 specific adjustments to optimize the learning environment."
        ),
        "2c": (
            "Assess the clarity and consistency of behavioral expectations communicated by the teacher. Look for well-defined routines and consistent enforcement of "
            "rules. Evaluate how the teacher's management strategies support a safe, orderly environment that is fair and conducive to learning. Provide 1-2 high-leverage "
            "suggestions for strengthening classroom procedures."
        ),
        "2d": (
            "Examine the teacher's use of proactive interventions to address potential academic or behavioral challenges. Look for evidence of targeted support for individual "
            "students, including differentiated interventions and timely adjustments based on ongoing assessments of student needs. Recommend 1-2 specific strategies "
            "to enhance student behavior management."
        ),
        "2e": (
            "Consider how both the physical and psychological aspects of the classroom environment impact learning. Assess elements such as lighting, seating, displays, "
            "and overall ambiance. Evaluate whether the space is intentionally organized to promote focus, comfort, and engagement. Suggest 1-2 actionable modifications "
            "to improve the physical environment."
        ),
        "3a": (
            "Evaluate the clarity and effectiveness of instructional delivery. Examine the teacher's communication style, pacing, and ability to present complex ideas in an accessible manner. "
            "Look for a balanced approach that combines direct instruction with interactive, student-centered activities. Provide 1-2 specific techniques to enhance communication effectiveness."
        ),
        "3b": (
            "Look for evidence of effective questioning techniques that stimulate higher-order thinking. Assess whether the teacher employs open-ended questions and probing prompts "
            "that encourage analysis, synthesis, and problem-solving. Evaluate how these methods foster student reflection and active engagement in the learning process. "
            "Recommend 1-2 questioning strategies to implement."
        ),
        "3c": (
            "Assess the teacher's adaptability in response to the diverse learning needs of students. Examine the use of real-time feedback, flexible grouping, and differentiated "
            "instructional strategies. Evaluate whether the teacher uses formative assessment data to make adjustments that enhance understanding for all learners. "
            "Suggest 1-2 specific approaches to improve responsiveness to student needs."
        ),
        "3d": (
            "Examine the integration of technology and multimedia resources into the lesson. Assess how digital tools and visual aids are used to enrich the instructional experience. "
            "Evaluate whether these resources are relevant to the lesson objectives and effectively enhance student engagement and comprehension. Recommend 1-2 actionable "
            "strategies for technology integration."
        ),
        "3e": (
            "Review the overall coherence of the lesson's structure and instructional materials. Assess whether the lesson flows logically, with each segment building upon previous knowledge. "
            "Evaluate the alignment of resources, activities, and assessments with the stated learning objectives to ensure a unified and effective learning experience. "
            "Provide 1-2 specific suggestions to strengthen lesson coherence."
        ),
        # Domain 4 typically is manually entered; add instructions if needed.
    }
    specific_instruction = component_instructions.get(
        component_id, "Focus on key evidence directly related to this component."
    )
    
    # Assemble the final prompt with context and observation details.
    prompt = (
        f"{base_prompt}{specific_instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Observation:\n{observation_text}\n\n"
        "Guidelines:\n"
        "1. Provide a JSON output with two keys: 'summary' and 'score'.\n"
        "2. 'score' must be an integer between 1 (Unsatisfactory) and 4 (Distinguished).\n"
        "3. 'summary' should include both:\n"
        "   - Performance analysis: What the teacher did well with specific evidence\n"
        "   - Growth path: 1-3 specific, actionable recommendations for improvement\n"
        "4. For each statement in your 'summary', include direct evidence from the observation.\n"
        "5. Do not include any commentary outside the JSON structure.\n"
    )
    return prompt

def generate_component_evaluation(component_id: str, observation_text: str, context: str) -> Dict[str, Any]:
    """
    Generate an evaluation for a specific Danielson component using a dynamic, component-specific prompt.
    This implementation uses:
      - A dynamically generated, concise prompt.
      - Component-specific instructions mapping to ensure the evaluation captures unique cues.
    
    Args:
        component_id (str): The Danielson component ID.
        observation_text (str): The observation text.
        context (str): Additional contextual analysis.
    
    Returns:
        Dict[str, Any]: JSON-formatted evaluation with keys 'summary' and 'score'.
    """
    # Build the dynamic prompt using our helper function.
    prompt = create_component_prompt(component_id, observation_text, context)
    
    try:
        response = generate_ai_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,  # Deterministic output; adjust if needed.
                response_mime_type="application/json"
            )
        )
        if response.parts:
            response_text = ''.join(part.text for part in response.parts).strip()
        else:
            return {"score": 1, "summary": ""}
        
        result = json.loads(response_text)
        result["score"] = normalize_score_integer(result.get("score", 1))
        return result
    except Exception as e:
        return {"score": 1, "summary": ""}

def normalize_score_integer(score: Any) -> int:
    """
    Convert any score value to an integer between 1 and 4.
    
    Args:
        score (Any): Input score value (can be string, float, int, or None)
        
    Returns:
        int: Normalized score between 1 and 4
    """
    try:
        float_score = float(score)
        int_score = round(float_score)
        return max(1, min(4, int_score))
    except (ValueError, TypeError):
        return 1

def restructure_component_feedback(text: str, evidence: str, component_id: str) -> str:
    """
    Restructure feedback for a single component using AI.
    
    Args:
        text (str): Original feedback text
        evidence (str): Original observation text containing evidence
        component_id (str): Component identifier (e.g., "1a")
        
    Returns:
        str: Restructured feedback
    """
    # List of high-quality evaluation examples to reference
    example_evaluations = [
        """There is clear student engagement through different instructional strategies; evidence supports mixed instructional strategies with a mix of independent work, sharing with a partner, and using student work to engage students in synthesis and analysis. You effectively used real-world context with sales tax to engage students, and you encouraged mathematical discourse when asking students to explain how they arrived at their answers. These practices help build foundational understanding. Continue to use discussion, visuals, and representations to help students understand the conceptual component of these standards.""",
        
        """The exploration of geometric concepts (attributes) maintains high levels of cognitive focus and participation. The pacing provides time for reflection and closure. Students are highly intellectually engaged through application of attributes to distinguish between two types of shapes and use this understanding to explain what makes a triangle a triangle and a circle a circle. The use of the document camera allows for student to student feedback, promotes critical thinking, and allows for synthesis of the content using a student example.""",
        
        """Students are intellectually engaged in analyzing geometric shapes and their attributes. They are actively involved in the lesson through participation and shared reasoning (i.e. students show agreement via a hand signal). Students were actively engaged in verbal mathematical discourse about shape attributes, particularly when you asked them to justify their thinking about the curved-sided triangle. This created a strong foundation for developing mathematical reasoning. As this is the synthesis of the lesson, be sure to continue to use your in-lesson data collection tool to identify which students can correctly identify attributes to describe shapes (i.e. 0 attributes, 1 attribute, 2 attributes) as a means to help form small group instruction for that day.""",
        
        """Learning tasks require student thinking and are aligned with instructional outcomes. The use of games and manipulatives promotes engagement. I noticed that while all students are working towards the same learning objectives, it is clear that the centers were differentiated for groups of students - some students worked with manipulatives, some worked with single digits, and others worked with double digits. Continue to use in-lesson data collection to form small groups for your daily lesson and consider ways in which students may have some choice in an center that aligns to their learning need.""",
        
        """Students appear intellectually engaged in meaningful work. The lesson provides multiple entry points for participation. An example of this is when engagement was fostered through partner discussions, evidenced by prompts like 'Turn and talk to a partner.' The use of a document camera to display student work and engage students in learning promotes higher-order thinking, critical feedback, analysis, and synthesis of learning. Use this opportunity to "stamp" the learning and bring learning back to the objective at the end of the synthesis.""",
        
        """Students are actively engaged through multiple modalities and meaningful tasks. The 'turn-and-talk' method suggests engagement, with students notably participating for 'bonus points.' Nonetheless, understanding engagement depth requires further information that you will need to adjust instruction along the way. This is the place that you will want to utilize in-lesson data collection to see who is on-track to meeting the learning objective and further understand their engagement and understanding of the lesson goals.""",
        
        """You effectively began with clear teacher modeling of proportional relationships and had students engage in partner work. Your questioning sequence showed intentional scaffolding from basic recall to more complex thinking about relationships between concepts. What do you notice about student independence levels during different parts of the lesson? How might students demonstrate they're ready to take on more responsibility in their learning? Continue to incorporate opportunities for discourse, utilizing student work to support students' ability to explain their thinking and solve similar problems independently."""
    ]

    prompt = f"""
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

    Common Themes from Exemplar Evaluations to Incorporate:
    1. Focus on student engagement through multiple modalities
    2. Emphasis on mathematical discourse and reasoning
    3. Use of technology (document cameras) and manipulatives
    4. Connection to real-world contexts
    5. Importance of in-lesson data collection
    6. Balance of individual, partner, and whole group work
    7. Attention to differentiation and multiple entry points
    8. Integration of student work examples
    9. Emphasis on synthesis and closure
    10. Progression from basic recall to complex thinking

    You should output it in Markdown format.
    Always start with **Performance Analysis** and then **Growth Path**. Always separate the sections with a new line; they are their own paragraphs.

    Use the following exemplar evaluations as a guide for your response:
    {example_evaluations}

    Your longer, improved component summary:
    """
    
    try:
        response = generate_ai_content(prompt)
        if response.parts:
            # Join all parts of the response
            text = ''.join(part.text for part in response.parts)
        else:
            return text  # Return original text if processing fails
        return text
    except Exception as e:
        return text  # Return original text if processing fails