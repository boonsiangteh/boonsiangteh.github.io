---
layout: default
title: Spring Boot – Practical Mental Model & Reference Notes
nav_order: 2
parent: Docs
---

## Spring Boot – Practical Mental Model & Reference Notes

This document captures the essential concepts behind Spring Boot and the surrounding Spring ecosystem.
The goal is not to memorize annotations, but to understand how requests flow, how objects are created, and why Spring exists in the first place.

---

## 1. What Spring Boot Is (and What It Is Not)

### What

Spring Boot is an opinionated framework built on top of the Spring Framework that helps you create production-ready Java/Kotlin applications quickly.

It provides:
- Auto-configuration
- Embedded servers (Tomcat, Jetty, Netty)
- Sensible defaults
- Minimal boilerplate

### What It Is Not

- It is not a web server itself.
- It is not an ORM.
- It does not replace Spring Core.

Spring Boot simply wires things together for you, so you don’t have to.

---

## 2. High-Level Request Flow (Big Picture)

```
HTTP Request
   |
   v
Servlet Container (Tomcat)
   |
   v
DispatcherServlet (Front Controller)
   |
   v
@Controller method
   |
   v
@Service (business logic)
   |
   v
@Repository / JPA Repository
   |
   v
Database
```

Understanding this flow explains 90% of Spring Boot behavior.

---

## 3. Servlet Containers & Servlets

### Servlet Container

**What**

A Servlet Container (e.g. Tomcat) is a runtime that:
- Listens on ports
- Accepts HTTP requests
- Manages servlet lifecycle

**Why**

Java web applications need a standardized environment to handle HTTP.
The servlet container provides this environment.

**How**

Spring Boot embeds Tomcat by default:

```
Spring Boot App
 └── Embedded Tomcat
     └── Servlet API
```

You don’t deploy WAR files anymore; you run a JAR.

### Servlets

**What**

A Servlet is a Java class that handles HTTP requests.

**Why**

They provide a low-level, standardized API for HTTP handling.

**How**

Spring hides raw servlets behind abstractions.
You rarely write servlets yourself.

---

## 4. DispatcherServlet (Front Controller Pattern)

**What**

`DispatcherServlet` is Spring MVC’s front controller.

**Why**

Instead of many servlets handling requests independently, Spring routes all requests through one place.

**How**

- Every HTTP request hits `DispatcherServlet`.
- It finds the correct controller method.
- It handles argument binding, validation, serialization.

You never create it manually—Spring Boot auto-configures it.

---

## 5. ApplicationContext

**What**

`ApplicationContext` is Spring’s container of objects (beans).

**Why**

Applications need a controlled way to:
- Create objects
- Manage lifecycles
- Inject dependencies

**How**

At startup:
1. Spring scans your classpath.
2. Finds annotated classes.
3. Creates beans.
4. Stores them in the `ApplicationContext`.

```
ApplicationContext
 ├── Controller bean
 ├── Service bean
 ├── Repository bean
 └── Other infrastructure beans
```

---

## 6. Beans

**What**

A bean is an object managed by Spring.

**Why**

Spring can:
- Control lifecycle
- Inject dependencies
- Apply cross-cutting concerns (transactions, security, logging)

**How**

Beans are created via:
- Component scanning
- Configuration classes
- Auto-configuration

Most beans are singletons by default.

---

## 7. Dependency Injection (DI)

**What**

Dependency Injection means objects do not create their own dependencies.

**Why**

- Loose coupling
- Easier testing
- Clear object responsibilities

**How**

Spring injects dependencies via:
- Constructor injection (recommended)
- Field injection (discouraged)
- Setter injection (rare)

**Example (Constructor Injection)**

```kotlin
@Service
class OrderService(
    private val paymentService: PaymentService
)
```

Spring:
- Creates `PaymentService`
- Passes it into `OrderService`

---

## 8. Core Stereotype Annotations

These annotations mark roles in your application.

### `@Controller`

**What**

Marks a class as a web controller.

**Why**

Spring needs to know which classes handle HTTP requests.

**How**

- Used with `@RequestMapping`, `@GetMapping`, etc.
- Returns responses (JSON, views, etc.)

```kotlin
@Controller
class UserController {

    @GetMapping("/users")
    fun listUsers(): List<String> {
        return listOf("Alice", "Bob")
    }
}
```

### `@Service`

**What**

Marks a class as business logic.

**Why**

Separates business rules from HTTP and persistence concerns.

**How**

Spring treats it as a bean and allows:
- Transactions
- Reuse across controllers

```kotlin
@Service
class UserService {
    fun processUser() {}
}
```

### `@Repository`

**What**

Marks a class responsible for data access.

**Why**

- Clear architectural boundary
- Exception translation (DB → Spring exceptions)

**How**

Usually used for custom data access logic.

### JPA Repository

**What**

Spring Data JPA provides repository interfaces with CRUD methods.

**Why**

Eliminates boilerplate DAO code.

**How**

You define an interface, Spring generates implementation at runtime.

```kotlin
interface UserRepository : JpaRepository<User, Long>
```

You get:
- `save`
- `findById`
- `findAll`
- `delete`

For free.

---

## 9. Validation

**What**

Bean Validation checks request data against constraints.

**Why**

Protects your system from invalid input.

**How**

- Use `@Valid` in controllers.
- Use constraint annotations on DTO fields.

```kotlin
data class CreateUserRequest(
    @field:NotBlank
    val name: String
)
```

Spring:
- Validates automatically
- Returns 400 Bad Request on failure

---

## 10. Unit Testing in Spring

**What**

Unit tests verify one class in isolation.

**Why**

- Fast
- Precise
- Easy to debug

**How**

- No Spring context
- Dependencies are mocked

---

## 11. MockK

**What**

MockK is a Kotlin-first mocking library.

**Why**

Mockito struggles with Kotlin features.

**How**

```kotlin
val repo = mockk<UserRepository>()
every { repo.findAll() } returns emptyList()
```

MockK:
- Creates fake implementations
- Controls behavior

---

## 12. MockMvc

**What**

MockMvc simulates HTTP requests without starting a server.

**Why**

- Test controllers realistically
- Fast execution

**How**

MockMvc sends fake HTTP requests to Spring MVC:

```kotlin
mockMvc.perform(get("/users"))
    .andExpect(status().isOk)
```

---

## 13. Integration Testing in Spring

**What**

Integration tests load the Spring context and test components working together.

**Why**

- Catch wiring issues
- Validate configuration
- Test real behavior

**How**

Spring Boot spins up:
- `ApplicationContext`
- Beans
- Embedded infrastructure

### Common Integration Test Annotations

- `@SpringBootTest` — Loads the full application context.
- `@AutoConfigureMockMvc` — Provides `MockMvc` with real Spring wiring.
- `@ActiveProfiles("test")` — Uses test-specific configuration.
- `@ExtendWith(SpringExtension::class)` — Integrates JUnit with Spring.

### Typical Integration Test Setup

```kotlin
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
class UserIntegrationTest {

    @Autowired
    lateinit var mockMvc: MockMvc
}
```

This:
- Loads the real app
- Uses test config
- Allows HTTP-level testing

---

## 14. Mental Model Summary

If you remember only this:
- Spring Boot starts the system
- Tomcat accepts requests
- DispatcherServlet routes requests
- Controllers handle HTTP
- Services contain logic
- Repositories access data
- ApplicationContext manages everything
- Tests choose how much of Spring to load

You understand Spring Boot.

---

## 15. When Things Go Wrong

Most Spring issues fall into one of these buckets:
- Bean not found → component scanning / configuration issue
- Validation not triggered → missing `@Valid`
- Test slow → loading too much context
- Runtime wiring errors → integration test needed

Spring is predictable once the mental model is clear.

---

## Final Thought

Spring Boot is less about magic and more about conventions plus a container.
Once you know who creates objects, who owns lifecycles, and how requests flow, the framework becomes boring—in a good way.

And boring systems are reliable systems.
