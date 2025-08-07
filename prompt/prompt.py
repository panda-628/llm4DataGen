#随机生成领域
PROMPT_GEN_DOMAIN = """
I want to create a domain model. Could you help me randomly select a concrete application domain, which will be used to fill in the subsequent <model skeleton>? 
For example: online bookstore.
To avoid redundancy, do not select the <domains you have generated>. 
You only need to give the name of the domain in following format:
#domain name
...

#model skeleton
{}

#domains you have generated
{}

"""

#AI对模型进行填充
PROMPT_GEN_MODEL = """
You are an assistant helping to generate domain models.
Please create a domain model in JSON format based on the provided <domain> and <model skeleton>.
Key Requirements:
1. Complete Mapping: Ensure that all identifiers, attributes, types, operations, and relationNames in the <model skeleton> are mapped ——no field should be omitted.
2. Domain Fitness: The generated content must closely align with the business domain specified in <domain>, with attributes and operations conforming to the logical characteristics of that domain.
3. Structural Integrity: Do NOT modify the provided <model skeleton> structure, including:
   - Class/entity names and their hierarchical organization
   - Existing attributes, operations, and their order
   - Defined relationships (types, multiplicities, names)
Ensure that the skeleton of the generative model is consistent with the given skeleton.
#domain: 
{}
#model skeleton:
{}
"""

#AI对模型进行映射
PROMPT_REPLACE_MODEL = """
You are a helpful assistant. Your task is to replace the <generated mapping> relationship with the <original model>.
For example, if the mapping relationship is:
"identifier1": "Smartphone", You should replace "identifier1" with "Smartphone".
Please provide the replaced model in plantUML format.

#original model
{}
#generated mapping
{}
"""

#AI对模型进行验证
PROMPT_VERIFY_MODEL = """
You are a helpful assistant. Your task is to verify whether the generated model is consistent with the given domain and to check if there are any unreasonable parts in the <generated model>.
Please provide a detailed explanation of the verification process and the results.
If the <generated model> does not conform to the given domain or there are unreasonable parts in the model, please make corrections and return the corrected model. You can change the model skeleton to make it more reasonable.
If it is consistent with the given <domain> and there are no unreasonable parts, return to the <generated model>

Please return the result in the following format:
#corrected model(use the same format as the <generated model>)
...
#verification result
...

#generated model
{}
#domain
{}
"""

#对模型生成描述
PROMPT_GEN_MODEL_DESCRIPTION = """
Generate the corresponding natural language description based on the given <domain model> and <domain>. Note: The given natural language description should be comprehensive and brief and presented in plain text format.
Note: 
1. Do not directly present the methods in the domain model in the generated system description, such as: "deleteName()", "ModifyPlan()".
2. Do not directly introduce the identifiers in the domain system into the system description.
Here is an example of a CelO system description:
description:
    The CelO application helps families and groups of friends to organize birthday celebrations and other events. 
    Organizers can keep track of which tasks have been completed and who attends. Attendees can indicate what they are bringing to the event. For a small event, there is typically one organizer, but larger events require several organizers. An organizer provides their first and last name, their email address :which is also used as their username, their postal address, their phone number, and their password. Furthermore, an organizer indicates the kind of event that needs to be planned by selecting from a list of events :e.g., birthday party, graduation party… or creating a new kind of event. The start date/time and end date/time of the event must be specified as well as the occasion and location of the event. 
    The location can again be selected from a list, or a new one can be created by specifying the name of the location and its address. An organizer then invites the attendees by entering their first and last names as well as their email addresses. Sometimes, an organizer is only managing the event but not attending the event. Sometimes, an organizer also attends the event. When an attendee receives the email invitation, the attendee can create an account :if they do not yet have an account with a new password and their email address from the invitation as their username. Afterwards, the attendee can indicate whether they will attend the event, maybe will attend the event, or cannot attend the event. An organizer can view the invitation status of an event, e.g., how many attendees have replied or have not yet replied and who is coming for sure or maybe will be coming. When an organizer selects an event, an event-specific checklist is presented to the organizer. For example, a birthday party may have a task to bring a birthday cake. For each task on the checklist, an organizer can indicate that the task needs to be done, has been done, or is not applicable for the event. An organizer can also add new tasks to the list, which will then also be available for the next event. For example, an organizer can add to bring birthday candles to the list for a birthday party and this task will then be available for the next birthday party, too. An organizer can also designate a task on the checklist for attendees to accomplish. For example, an organizer can indicate that the birthday cake should be brought to the event by an attendee. If this is the case, then the list of tasks to be accomplished by attendees is shown to attendees that have confirmed their attendance to the event. An attendee can then select their tasks, so that the organizer can see who is bringing"
Please provide the description in the following format:
#System description
...

#domain model
{}
#domain
{}
"""
# PROMPT_GEN_MODEL_DESCRIPTION = """
# Generate the corresponding natural language description based on the given <domain model> and <domain>. Note: The given natural language description should conform to natural language spelling and grammar, and the description should be comprehensive and brief.
# #domain model
# {}
# #domain
# {}
# Here is an example of a domain model and its description:
# Oracle model:
# @startuml
# enum TaskType {
#   NEEDED
#   DONE
#   NOT_APPLICABLE
# }

# enum InvitationStatus {
#   NO_RESPONSE
#   ATTENDING
#   MAYBE
#   NOT_ATTENDING
# }
# class Event {
#   -name: String
#   -startDateTime: Date
#   -endDateTime: Date
#   -occasion: String
# }

# class Invitation {
#   -firstName: String
#   -lastName: String
#   -emailAddress: String
#   -status: InvitationStatus
# }

# class Location {
#   -name: String
#   -address: String
# }

# class Task {
#   -taskType: TaskType
#   -description: String
#   -isAttendeeTask: boolean
# }

# class User {
#   -firstName: String
#   -lastName: String
#   -emailAddress: String
#   -password: String
#   -phoneNumber: String
#   -postalAddress: String
# }

# class EventKind {
#   -name: String
# }

# class TaskTemplate {
#   -taskType: TaskType
#   -description: String
#   -isAttendeeTask: boolean
# }
# Event "*" -- "*" Invitation : invitations
# Event "1" -- "1" Location : location
# Event "*" -- "*" Task : checklist
# Invitation "1" -- "0..1" User : invitee
# Task "1" -- "1" User : accomplisher
# User "*" -- "*" Event : organizers
# Event "*" -- "*" EventKind : Kind
# User "*" -- "*" EventKind : EventKinds
# EventKind "*" -- "*" TaskTemplate : taskTemplates

# @enduml
# description:
#     The CelO application helps families and groups of friends to organize birthday celebrations and other events. 
#     Organizers can keep track of which tasks have been completed and who attends. Attendees can indicate what they are bringing to the event. For a small event, there is typically one organizer, but larger events require several organizers. An organizer provides their first and last name, their email address :which is also used as their username, their postal address, their phone number, and their password. Furthermore, an organizer indicates the kind of event that needs to be planned by selecting from a list of events :e.g., birthday party, graduation party… or creating a new kind of event. The start date/time and end date/time of the event must be specified as well as the occasion and location of the event. 
#     The location can again be selected from a list, or a new one can be created by specifying the name of the location and its address. An organizer then invites the attendees by entering their first and last names as well as their email addresses. Sometimes, an organizer is only managing the event but not attending the event. Sometimes, an organizer also attends the event. When an attendee receives the email invitation, the attendee can create an account :if they do not yet have an account with a new password and their email address from the invitation as their username. Afterwards, the attendee can indicate whether they will attend the event, maybe will attend the event, or cannot attend the event. An organizer can view the invitation status of an event, e.g., how many attendees have replied or have not yet replied and who is coming for sure or maybe will be coming. When an organizer selects an event, an event-specific checklist is presented to the organizer. For example, a birthday party may have a task to bring a birthday cake. For each task on the checklist, an organizer can indicate that the task needs to be done, has been done, or is not applicable for the event. An organizer can also add new tasks to the list, which will then also be available for the next event. For example, an organizer can add to bring birthday candles to the list for a birthday party and this task will then be available for the next birthday party, too. An organizer can also designate a task on the checklist for attendees to accomplish. For example, an organizer can indicate that the birthday cake should be brought to the event by an attendee. If this is the case, then the list of tasks to be accomplished by attendees is shown to attendees that have confirmed their attendance to the event. An attendee can then select their tasks, so that the organizer can see who is bringing"
# """

#对生成的描述进行校验
PROMPT_VERIFY_MODEL_DESCRIPTION = """
You are a helpful assistant. Your task is to verify whether the generated description is consistent with the given <domain model> and to check if there are any unreasonable parts in the <generated description>.
Please provide a detailed explanation of the verification process and the results.
If the <generated description> does not conform to the given <domain model> or there are unreasonable parts in the description, please make corrections and return the corrected description.
Please provide the explanation of the verification process and thefinal modified description in the following format:
#explanation of the verification process
...
#Final modified description
...

#generated description
{}
#domain model
{}
"""