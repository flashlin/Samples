using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class __RefactorLogEntityConfiguration : IEntityTypeConfiguration<__RefactorLogEntity>
    {
        public void Configure(EntityTypeBuilder<__RefactorLogEntity> builder)
        {
            builder.ToTable("__RefactorLog");

            builder.HasKey(x => x.OperationKey);

            builder.Property(x => x.OperationKey)
                .HasColumnType("uniqueidentifier")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

        }
    }
}
