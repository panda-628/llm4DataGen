running_params = {
    'llm': 'deepseek-r1-250528',
    'API_KEY': 'sk-PtSmjHPGCTXQvXPBGQf7Av1tiHqYKRvDwExuSc3MJPVIhn1d',
    'temperature': 0.9,
    'max_tokens': 3000,
    'BASE_URL': 'https://www.dmxapi.com/v1',
    'cycles': 1,
    "embedding_threshold": 0.8,
}

file = {
    'path': 'C:\AppAndData\codeAndproject\llm4Data\labResult',
}

MODEL_FRAMEWORK = """
   @startuml
class identifier1 {
    - attribute1: identifier2
    - attribute2: List<identifier3>
    + identifier1(identifier2 attribute1, List<identifier3> attribute2)
    + operation1(): void
    + operation2(identifier4 attribute3): void
}

class identifier2 {
    + operation3(): void
}

class identifier3 {
    + operation4(): void
}

class identifier4 {
    + operation5(): void
}

class identifier5 {
    + operation3(): void
}

class identifier6 {
    + operation3(): void
}

class identifier7 {
    + operation4(): void
}

class identifier8 {
    + operation4(): void
}

class identifier9 {
    - attribute4: String
    + operation5(): void
    + operation6(): void
}

class identifier10 {
    - attribute5: String
    + operation5(): void
    + operation6(): void
}

identifier1 *-- identifier2
identifier1 *-- identifier3
identifier1 ..> identifier4
identifier2 <|-- identifier5
identifier2 <|-- identifier6
identifier3 <|-- identifier7
identifier3 <|-- identifier8
identifier4 <|-- identifier9
identifier4 <|-- identifier10
@enduml
"""