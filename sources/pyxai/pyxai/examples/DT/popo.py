# # Check V1.0: Ok
# from pyxai import Learning, Explainer, Tools
# name='balance-scale_0'
# # Machine learning part
# dt_learner = Learning.Scikitlearn(name+'.csv', learner_type=Learning.CLASSIFICATION)

# model = dt_learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
# instance, prediction = dt_learner.get_instances(model, n=1, correct=True)

# # Explanation part
# print(instance)
# # explainer = Explainer.decision_tree(model, instance)
# explainer = Explainer.initialize(model, instance, features_type= name+'.types')
# explainer.add_clause_to_theory([3, -4])
# print("instance:", instance)
# print("binary: ", explainer.binary_representation)
# reason = explainer.minimal_majoritary_reason()
# print("reason: ", reason)
# print("is reason", explainer.is_reason(reason))
# clauses=[]
# for clause in model.get_theory(explainer.binary_representation):
#     print("oo")
#     print(clause)


# name='balance-scale_0'
# # Check V1.0: Ok
# from pyxai import Learning, Explainer, Tools

# # Machine learning part
# learner = Learning.Scikitlearn(name+'.csv', learner_type=Learning.CLASSIFICATION)

# model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.DT)
# instance, prediction = learner.get_instances(model, n=1, correct=True)

# # Explanation part
# explainer = Explainer.decision_tree(model, instance)
# explainer.add_clause_to_theory([-3, 4])
# explainer.add_clause_to_theory([-2])
# print("instance:", instance)
# print("binary: ", explainer.binary_representation)
# reason = explainer.minimal_majoritary_reason()
# print("reason: ", reason)
# print("is reason", explainer.is_reason(reason))


name = '../../dataset/balance-scale_0_vs_1'

from pyxai import Learning, Explainer, Tools

# Machine learning part
learner = Learning.Scikitlearn(name + '.csv', learner_type=Learning.CLASSIFICATION)

model = learner.evaluate(method=Learning.HOLD_OUT, output=Learning.RF)
instance, prediction = learner.get_instances(model, n=1, correct=True)

# Explanation part
explainer = Explainer.initialize(model, instance, features_type=name + '.types')
#explainer.add_clause_to_theory([1, 2])
# explainer.add_clause_to_theory([2, -3])
explainer.add_clause_to_theory([-5, 9,13])
print(explainer.get_theory())
print("instance:", instance)
print("binary: ", explainer.binary_representation)

print(explainer.instance_compatible_with_theory())


if explainer.instance_compatible_with_theory():
    reason = explainer.majoritary_reason()
    print("reason: ", reason)
    print("tofeature", explainer.to_features(reason))
    print("is reason", explainer.is_reason(reason))
else :
    print("Instance non compatible with theory")